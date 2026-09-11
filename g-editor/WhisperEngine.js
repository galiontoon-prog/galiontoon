/* WhisperEngine.js · motor OPCIONAL. Misma carpeta que el HTML.
   API: cargar / transcribir / estado / borrar
   CDN (Hugging Face via transformers.js) SOLO al pulsar Descargar modelo.
   Cache: Cache API + marca OPFS. WebGPU → WASM. Audio no sale del equipo.
*/
(function (global) {
  'use strict';

  var MODELO = 'onnx-community/whisper-tiny';
  var TF_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2';
  var estado = 'no-descargado';
  var ultimoError = '';
  var pipe = null;
  var tfMod = null;
  var worker = null;
  var reqId = 0;
  var pendientes = Object.create(null);
  var onProgreso = null;
  var webgpu = typeof navigator !== 'undefined' && !!navigator.gpu;

  function postEstado() {
    try {
      global.dispatchEvent(new CustomEvent('whisper-estado', { detail: estado }));
    } catch (_) {}
  }

  function setEstado(s, err) {
    estado = s;
    if (err) ultimoError = String(err);
    postEstado();
  }

  function idiomaWhisper(code) {
    var c = String(code || 'es').toLowerCase().split(/[-_]/)[0];
    var map = {
      es: 'spanish', en: 'english', fr: 'french', de: 'german', it: 'italian',
      pt: 'portuguese', ca: 'catalan', gl: 'galician', eu: 'basque',
      ja: 'japanese', zh: 'chinese', ko: 'korean', ru: 'russian',
      ar: 'arabic', hi: 'hindi', nl: 'dutch', pl: 'polish', tr: 'turkish'
    };
    return map[c] || c;
  }

  async function marcarOPFS(listo) {
    try {
      if (!navigator.storage || !navigator.storage.getDirectory) return;
      var root = await navigator.storage.getDirectory();
      if (!listo) {
        try { await root.removeEntry('whisper-engine', { recursive: true }); } catch (_) {}
        return;
      }
      var d = await root.getDirectoryHandle('whisper-engine', { create: true });
      var f = await d.getFileHandle('ready.txt', { create: true });
      var w = await f.createWritable();
      await w.write(MODELO + '\n' + Date.now());
      await w.close();
    } catch (_) {}
  }

  async function hayMarcaOPFS() {
    try {
      if (!navigator.storage || !navigator.storage.getDirectory) return false;
      var root = await navigator.storage.getDirectory();
      var d = await root.getDirectoryHandle('whisper-engine', { create: false });
      await d.getFileHandle('ready.txt', { create: false });
      return true;
    } catch (_) { return false; }
  }

  async function hayCacheHF() {
    try {
      var ks = await caches.keys();
      return ks.some(function (k) { return /huggingface|transformers|whisper/i.test(k); });
    } catch (_) { return false; }
  }

  async function pcm16k(blob) {
    var AC = global.AudioContext || global.webkitAudioContext;
    var ac = new AC();
    var raw;
    try {
      raw = await blob.arrayBuffer();
    } catch (e) {
      try { ac.close(); } catch (_) {}
      throw e;
    }
    var buf;
    try {
      buf = await ac.decodeAudioData(raw.slice(0));
    } finally {
      try { ac.close(); } catch (_) {}
    }
    var n = buf.length, chs = buf.numberOfChannels;
    var mono = new Float32Array(n);
    for (var c = 0; c < chs; c++) {
      var ch = buf.getChannelData(c);
      for (var i = 0; i < n; i++) mono[i] += ch[i] / chs;
    }
    var sr = buf.sampleRate;
    if (sr === 16000) return mono;
    var ratio = sr / 16000;
    var outLen = Math.max(1, Math.round(n / ratio));
    var out = new Float32Array(outLen);
    for (var j = 0; j < outLen; j++) {
      var x = j * ratio;
      var i0 = Math.floor(x);
      var i1 = Math.min(n - 1, i0 + 1);
      var f = x - i0;
      out[j] = mono[i0] * (1 - f) + mono[i1] * f;
    }
    return out;
  }

  function segsDeResultado(r) {
    var out = [];
    if (!r) return out;
    if (Array.isArray(r.chunks) && r.chunks.length) {
      r.chunks.forEach(function (ch) {
        var ts = ch.timestamp || [0, 0];
        var t0 = +ts[0] || 0;
        var t1 = ts[1] == null ? t0 + 1 : +ts[1];
        var texto = String(ch.text || '').trim();
        if (texto && t1 > t0) out.push({ t0: t0, t1: t1, texto: texto });
      });
      return out;
    }
    var txt = String(r.text || '').trim();
    if (txt) out.push({ t0: 0, t1: 1, texto: txt });
    return out;
  }

  function workerSrc() {
    return [
      'import { pipeline, env } from "' + TF_URL + '";',
      'env.allowLocalModels = false;',
      'env.useBrowserCache = true;',
      'let pipe = null;',
      'self.onmessage = async (ev) => {',
      '  const { id, tipo, payload } = ev.data || {};',
      '  try {',
      '    if (tipo === "cargar") {',
      '      const device = payload.webgpu ? "webgpu" : "wasm";',
      '      pipe = await pipeline("automatic-speech-recognition", payload.model, {',
      '        device,',
      '        dtype: payload.webgpu ? "fp16" : "q8",',
      '        progress_callback: (p) => self.postMessage({ id, tipo: "progreso", p })',
      '      });',
      '      self.postMessage({ id, tipo: "ok" });',
      '    } else if (tipo === "transcribir") {',
      '      const r = await pipe(payload.audio, {',
      '        language: payload.language,',
      '        task: "transcribe",',
      '        return_timestamps: true,',
      '        chunk_length_s: 30',
      '      });',
      '      self.postMessage({ id, tipo: "ok", result: r });',
      '    } else {',
      '      self.postMessage({ id, tipo: "error", msg: "tipo desconocido" });',
      '    }',
      '  } catch (e) {',
      '    self.postMessage({ id, tipo: "error", msg: String(e && e.message || e) });',
      '  }',
      '};'
    ].join('\n');
  }

  function llamarWorker(tipo, payload, transfer) {
    return new Promise(function (ok, mal) {
      var id = ++reqId;
      pendientes[id] = { ok: ok, mal: mal };
      try {
        worker.postMessage({ id: id, tipo: tipo, payload: payload }, transfer || []);
      } catch (e) {
        delete pendientes[id];
        mal(e);
      }
    });
  }

  function arrancarWorker() {
    if (worker) return;
    var blob = new Blob([workerSrc()], { type: 'text/javascript' });
    var url = URL.createObjectURL(blob);
    worker = new Worker(url, { type: 'module' });
    worker.onmessage = function (ev) {
      var d = ev.data || {};
      if (d.tipo === 'progreso') {
        if (typeof onProgreso === 'function') {
          var p = d.p || {};
          var pct = typeof p.progress === 'number' ? p.progress : (p.status === 'done' ? 100 : 0);
          onProgreso({ stage: p.status || '', file: p.file || '', progress: pct });
        }
        return;
      }
      var pend = pendientes[d.id];
      if (!pend) return;
      delete pendientes[d.id];
      if (d.tipo === 'ok') pend.ok(d.result);
      else pend.mal(new Error(d.msg || 'error Whisper'));
    };
    worker.onerror = function (e) {
      setEstado('error', e.message || 'worker');
    };
  }

  async function cargarEnHilo(opts) {
    tfMod = await import(TF_URL);
    tfMod.env.allowLocalModels = false;
    tfMod.env.useBrowserCache = true;
    var device = webgpu ? 'webgpu' : 'wasm';
    try {
      pipe = await tfMod.pipeline('automatic-speech-recognition', MODELO, {
        device: device,
        dtype: webgpu ? 'fp16' : 'q8',
        progress_callback: function (p) {
          if (opts && opts.onProgreso) {
            var pct = typeof p.progress === 'number' ? p.progress : 0;
            opts.onProgreso({ stage: p.status || '', file: p.file || '', progress: pct });
          }
        }
      });
    } catch (e) {
      if (device === 'webgpu') {
        pipe = await tfMod.pipeline('automatic-speech-recognition', MODELO, {
          device: 'wasm',
          dtype: 'q8',
          progress_callback: function (p) {
            if (opts && opts.onProgreso) {
              opts.onProgreso({ stage: p.status || '', file: p.file || '', progress: p.progress || 0 });
            }
          }
        });
      } else {
        throw e;
      }
    }
  }

  async function cargar(opciones) {
    opciones = opciones || {};
    onProgreso = opciones.onProgreso || opciones.onProgress || null;
    if (estado === 'listo' && (pipe || worker)) return { ok: true };
    setEstado('descargando');
    try {
      var usarWorker = opciones.worker !== false;
      if (usarWorker) {
        try {
          arrancarWorker();
          await llamarWorker('cargar', { model: MODELO, webgpu: webgpu });
          setEstado('listo');
          await marcarOPFS(true);
          return { ok: true, via: 'worker', webgpu: webgpu };
        } catch (e) {
          try { worker.terminate(); } catch (_) {}
          worker = null;
        }
      }
      await cargarEnHilo(opciones);
      setEstado('listo');
      await marcarOPFS(true);
      return { ok: true, via: 'main', webgpu: webgpu };
    } catch (e) {
      setEstado('error', e.message || e);
      throw e;
    }
  }

  async function transcribir(blob, opciones) {
    opciones = opciones || {};
    if (estado !== 'listo') {
      await cargar({ onProgreso: onProgreso });
    }
    if (!(blob instanceof Blob)) throw new Error('transcribir espera un Blob de audio');
    var audio = await pcm16k(blob);
    var lang = idiomaWhisper(opciones.idioma || opciones.language || 'es');
    var r;
    if (worker) {
      r = await llamarWorker('transcribir', { audio: audio, language: lang }, [audio.buffer]);
    } else if (pipe) {
      r = await pipe(audio, {
        language: lang,
        task: 'transcribe',
        return_timestamps: true,
        chunk_length_s: 30
      });
    } else {
      throw new Error('Whisper no está listo');
    }
    return { segmentos: segsDeResultado(r), bruto: r };
  }

  async function borrar() {
    try { if (worker) worker.terminate(); } catch (_) {}
    worker = null;
    pipe = null;
    tfMod = null;
    try {
      var ks = await caches.keys();
      for (var i = 0; i < ks.length; i++) {
        if (/huggingface|transformers|whisper/i.test(ks[i])) await caches.delete(ks[i]);
      }
    } catch (_) {}
    await marcarOPFS(false);
    setEstado('no-descargado');
    return { ok: true };
  }

  (async function probe() {
    try {
      if (await hayMarcaOPFS() || await hayCacheHF()) {
        /* Caché en disco, todavía no en memoria. */
        setEstado('no-descargado');
      }
    } catch (_) {}
  })();

  global.WhisperEngine = {
    cargar: cargar,
    transcribir: transcribir,
    borrar: borrar,
    estado: function () { return estado; },
    error: function () { return ultimoError; },
    webgpu: function () { return webgpu; },
    modelo: function () { return MODELO; },
    hayCache: async function () { return (await hayMarcaOPFS()) || (await hayCacheHF()); }
  };
})(typeof window !== 'undefined' ? window : self);
