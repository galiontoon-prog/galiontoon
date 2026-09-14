/* WhisperEngine.js · motor OPCIONAL. Misma carpeta que el HTML.
   API: cargar / transcribir / estado / borrar
   CDN (Hugging Face via transformers.js) SOLO al pulsar Descargar modelo.
   Cache: Cache API + marca OPFS. WebGPU → WASM. Audio no sale del equipo.
*/
(function (global) {
  'use strict';

  var TAMANO = 'tiny';
  var MODELOS = {
    tiny: 'onnx-community/whisper-tiny',
    base: 'onnx-community/whisper-base'
  };
  var MODELO = MODELOS.tiny;
  var CORS_PROXY = 'https://galiontoon-cors.galiontoon.workers.dev/?url=';
  var RELEASE_WH = 'https://github.com/galiontoon-prog/galiontoon/releases/download/modelos-whisper-v1/';
  var SILERO_NOM = 'silero_vad.onnx';
  var TF_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2';
  var TF_DIST = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2/dist/';
  var HF_HOST = 'https://huggingface.co/';

  function aplicarEnv(env) {
    if (!env) return;
    env.allowLocalModels = false;
    env.allowRemoteModels = true;
    env.useBrowserCache = true;
    env.remoteHost = HF_HOST;
    env.remotePathTemplate = '{model}/resolve/{revision}/';
    try {
      if (env.backends && env.backends.onnx && env.backends.onnx.wasm) {
        env.backends.onnx.wasm.wasmPaths = TF_DIST;
      }
    } catch (_) {}
  }
  var estado = 'no-descargado';
  var ultimoError = '';
  var pipe = null;
  var tfMod = null;
  var worker = null;
  var reqId = 0;
  var pendientes = Object.create(null);
  var onProgreso = null;
  var webgpu = typeof navigator !== 'undefined' && !!navigator.gpu;
  var webgpuOk = false;
  var deviceUsado = 'wasm';
  var dtypeUsado = 'q8';
  var idiomaSesion = '';
  var cargaEnCurso = null;

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

  async function dirMotor(crear) {
    if (!navigator.storage || !navigator.storage.getDirectory) return null;
    var root = await navigator.storage.getDirectory();
    return root.getDirectoryHandle('whisper-engine', { create: !!crear });
  }

  async function marcarOPFS(listo) {
    try {
      if (!listo) {
        if (!navigator.storage || !navigator.storage.getDirectory) return;
        var root = await navigator.storage.getDirectory();
        try { await root.removeEntry('whisper-engine', { recursive: true }); } catch (_) {}
        return;
      }
      var d = await dirMotor(true);
      if (!d) return;
      var f = await d.getFileHandle('ready.txt', { create: true });
      var w = await f.createWritable();
      await w.write(MODELO + '\n' + Date.now());
      await w.close();
    } catch (_) {}
  }

  async function hayMarcaOPFS() {
    try {
      var d = await dirMotor(false);
      if (!d) return false;
      await d.getFileHandle('ready.txt', { create: false });
      return true;
    } catch (_) { return false; }
  }

  async function leerLangs() {
    try {
      var d = await dirMotor(false);
      if (!d) return {};
      var f = await d.getFileHandle('langs.json', { create: false });
      var file = await f.getFile();
      var j = JSON.parse(await file.text());
      return j && typeof j === 'object' ? j : {};
    } catch (_) { return {}; }
  }

  async function escribirLangs(map) {
    var d = await dirMotor(true);
    if (!d) return;
    var f = await d.getFileHandle('langs.json', { create: true });
    var w = await f.createWritable();
    await w.write(JSON.stringify(map || {}));
    await w.close();
  }

  async function marcarIdioma(code, extra) {
    var k = String(code || 'auto').toLowerCase();
    var map = await leerLangs();
    map[k] = Object.assign({ ts: Date.now(), bytes: 75 * 1024 * 1024, modelo: MODELO }, extra || {});
    await escribirLangs(map);
    await marcarOPFS(true);
    return map[k];
  }

  async function idiomaListo(code) {
    var k = String(code || 'auto').toLowerCase();
    var map = await leerLangs();
    if (map[k]) return true;
    if (k === 'auto') {
      var keys = Object.keys(map);
      if (keys.length) return true;
      return (await hayMarcaOPFS()) || (await hayCacheHF());
    }
    return false;
  }

  async function borrarIdioma(code) {
    var k = String(code || '').toLowerCase();
    var map = await leerLangs();
    delete map[k];
    await escribirLangs(map);
    if (!Object.keys(map).length) await borrar();
    return { ok: true, quedan: Object.keys(map) };
  }

  async function bytesCache() {
    var total = 0;
    try {
      var ks = await caches.keys();
      for (var i = 0; i < ks.length; i++) {
        if (!/huggingface|transformers|whisper|gs-whisper/i.test(ks[i])) continue;
        var c = await caches.open(ks[i]);
        var reqs = await c.keys();
        for (var j = 0; j < reqs.length; j++) {
          try {
            var r = await c.match(reqs[j]);
            if (!r) continue;
            var b = await r.blob();
            total += b.size || 0;
          } catch (_) {}
        }
      }
    } catch (_) {}
    return total;
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

  var sesSilero = null;
  var sileroListo = false;
  async function probeWebGPU() {
    webgpuOk = false;
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      sileroLog('INFO sin navigator.gpu, WASM');
      return false;
    }
    try {
      var ad = await navigator.gpu.requestAdapter();
      if (!ad) {
        sileroLog('INFO requestAdapter()=null, WASM');
        return false;
      }
      webgpuOk = true;
      var inf = ad.info || {};
      sileroLog('INFO adapter', inf.vendor || '', inf.architecture || '', inf.isFallbackAdapter ? 'fallback' : '');
      return true;
    } catch (e) {
      sileroLog('INFO probe WebGPU', e && e.message || e);
      return false;
    }
  }
  function tunarWasm(ort) {
    if (!ort || !ort.env || !ort.env.wasm) return;
    try {
      var w = ort.env.wasm;
      w.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
      w.simd = true;
      var aislado = typeof crossOriginIsolated !== 'undefined' && !!crossOriginIsolated;
      var cores = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 1;
      w.numThreads = aislado ? Math.min(4, cores) : 1;
      w.proxy = !!aislado;
      sileroLog('WASM', 'simd', !!w.simd, 'threads', w.numThreads, 'proxy', w.proxy, 'COOP/COEP', aislado);
    } catch (e) {
      sileroLog('INFO tunar WASM', e && e.message || e);
    }
  }
  async function asegurarOrt() {
    if (global.ort) {
      tunarWasm(global.ort);
      return global.ort;
    }
    await new Promise(function (ok, bad) {
      var s = document.createElement('script');
      s.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js';
      s.onload = ok;
      s.onerror = function () { bad(new Error('ORT')); };
      document.head.appendChild(s);
    });
    tunarWasm(global.ort);
    return global.ort;
  }
  function sileroLog() {
    try { console.info.apply(console, ['[Silero]'].concat([].slice.call(arguments))); } catch (_) {}
  }
  async function asegurarSilero(onProg) {
    if (sesSilero) return sesSilero;
    var buf = null;
    var url = CORS_PROXY + RELEASE_WH + SILERO_NOM;
    try {
      var root = await navigator.storage.getDirectory();
      var d = await root.getDirectoryHandle('whisper-engine', { create: true });
      try {
        var fh = await d.getFileHandle(SILERO_NOM, { create: false });
        var file = await fh.getFile();
        if (file && file.size > 1024) {
          buf = await file.arrayBuffer();
          sileroLog('OPFS', file.size, 'bytes');
        }
      } catch (e) { sileroLog('OPFS vacío', e && e.message); }
      if (!buf) {
        sileroLog('intentando cargar', url);
        if (onProg) onProg({ file: SILERO_NOM, progress: 10 });
        var res = await fetch(url, { mode: 'cors' });
        sileroLog('HTTP status:', res.status, 'length', res.headers.get('content-length'));
        if (!res.ok) throw new Error('Silero HTTP ' + res.status);
        buf = await res.arrayBuffer();
        sileroLog('tamaño:', buf.byteLength, 'bytes');
        if (buf.byteLength < 10000) throw new Error('archivo Silero demasiado pequeño');
        var w = await (await d.getFileHandle(SILERO_NOM, { create: true })).createWritable();
        await w.write(buf); await w.close();
      }
    } catch (e) {
      sileroLog('ERROR descarga', e && e.message || e);
      throw e;
    }
    var ort = await asegurarOrt();
    var gpu = await probeWebGPU();
    var providers = gpu ? ['webgpu', 'wasm'] : ['wasm'];
    var lastOrt = null;
    for (var pi = 0; pi < providers.length; pi++) {
      try {
        sileroLog('sesión ONNX', providers[pi]);
        sesSilero = await ort.InferenceSession.create(buf, { executionProviders: [providers[pi]] });
        lastOrt = null;
        sileroListo = true;
        try { global.sileroListo = true; } catch (_) {}
        sileroLog('sesión ONNX creada OK', providers[pi], sesSilero.inputNames, sesSilero.outputNames);
        sileroLog('marcado como listo para el pipeline');
        break;
      } catch (e) {
        lastOrt = e;
        var msg = String(e && e.message || e);
        if (/webgpu|backend|not available/i.test(msg) && providers[pi] === 'webgpu')
          sileroLog('INFO WebGPU no disponible, usando WASM');
        else
          sileroLog('ERROR sesión', providers[pi], msg);
      }
    }
    if (!sesSilero) throw lastOrt || new Error('Silero session');
    return sesSilero;
  }
  async function vadSilero(blob) {
    var pcm = await pcm16k(blob);
    try { await asegurarSilero(); }
    catch (e) { sileroLog('ERROR init', e && e.message || e); return null; }
    var ort = global.ort;
    var hop = 512, sr = 16000, th = 0.45;
    var probs = [];
    var names = sesSilero.inputNames || ['input', 'state', 'sr'];
    var outs = sesSilero.outputNames || ['output', 'stateN'];
    var state = new ort.Tensor('float32', new Float32Array(2 * 1 * 128), [2, 1, 128]);
    var srTen = new ort.Tensor('int64', BigInt64Array.from([BigInt(sr)]), [1]);
    var falloRun = null;
    for (var i = 0; i + hop <= pcm.length; i += hop) {
      var sl = new Float32Array(hop);
      sl.set(pcm.subarray(i, i + hop));
      var feeds = {};
      for (var ni = 0; ni < names.length; ni++) {
        var nm = names[ni];
        if (/sr/i.test(nm)) feeds[nm] = srTen;
        else if (/state/i.test(nm)) feeds[nm] = state;
        else feeds[nm] = new ort.Tensor('float32', sl, [1, hop]);
      }
      try {
        var out = await sesSilero.run(feeds);
        var outKey = outs[0] && out[outs[0]] ? outs[0] : (out.output ? 'output' : Object.keys(out)[0]);
        var p = out[outKey] && out[outKey].data ? out[outKey].data[0] : 0;
        probs.push({ t: i / sr, p: +p });
        var stKey = outs.filter(function (n) { return /state/i.test(n); })[0];
        if (stKey && out[stKey]) state = out[stKey];
      } catch (e) {
        falloRun = e;
        sileroLog('ERROR inferencia', e && e.message || e);
        break;
      }
    }
    if (!probs.length) {
      sileroLog('sin frames', falloRun && (falloRun.message || falloRun));
      return null;
    }
    var segs = [], on = false, t0 = 0, last = 0;
    for (var k = 0; k < probs.length; k++) {
      var hit = probs[k].p >= th;
      if (!on && hit) { on = true; t0 = Math.max(0, probs[k].t - 0.05); }
      if (on && hit) last = probs[k].t + hop / sr;
      if (on && !hit && probs[k].t - last > 0.2) {
        if (last - t0 >= 0.35) segs.push({ t0: t0, t1: last, texto: '[voz]' });
        on = false;
      }
    }
    if (on && last - t0 >= 0.35) segs.push({ t0: t0, t1: last, texto: '[voz]' });
    return segs;
  }

  function limpiarTextoWh(s) {
    var t = String(s || '').replace(/\s+/g, ' ').trim();
    if (!t) return '';
    t = t.replace(/(.{2,4})\1{3,}/gi, '$1$1$1');
    var w = t.split(' ');
    var out = [];
    var run = 0, last = '';
    for (var i = 0; i < w.length; i++) {
      var cur = w[i];
      if (cur.toLowerCase() === last.toLowerCase()) {
        run++;
        if (run >= 3) continue;
      } else { run = 1; last = cur; }
      out.push(cur);
    }
    return out.join(' ').trim();
  }
  function segsDeResultado(r) {
    var out = [];
    if (!r) return out;
    function okMeta(ch) {
      if (!ch) return true;
      if (ch.no_speech_prob != null && +ch.no_speech_prob > 0.8) return false;
      if (ch.avg_logprob != null && +ch.avg_logprob < -1.5) return false;
      if (ch.compression_ratio != null && +ch.compression_ratio > 3) return false;
      return true;
    }
    if (Array.isArray(r.chunks) && r.chunks.length) {
      r.chunks.forEach(function (ch) {
        if (!okMeta(ch)) return;
        var ts = ch.timestamp || [0, 0];
        var t0 = +ts[0] || 0;
        var t1 = ts[1] == null ? t0 + 1 : +ts[1];
        var texto = limpiarTextoWh(ch.text || '');
        if (texto.length < 2) return;
        if (t1 - t0 < 0.5) return;
        out.push({ t0: t0, t1: t1, texto: texto });
      });
      return out;
    }
    var txt = limpiarTextoWh(r.text || '');
    if (txt.length >= 2) out.push({ t0: 0, t1: 1, texto: txt });
    return out;
  }

  function workerSrc() {
    return [
      'import { pipeline, env } from "' + TF_URL + '";',
      'env.allowLocalModels = false;',
      'env.allowRemoteModels = true;',
      'env.useBrowserCache = true;',
      'env.remoteHost = "' + HF_HOST + '";',
      'env.remotePathTemplate = "{model}/resolve/{revision}/";',
      'try { if (env.backends && env.backends.onnx && env.backends.onnx.wasm) env.backends.onnx.wasm.wasmPaths = "' + TF_DIST + '"; } catch (e) {}',
      'let pipe = null;',
      'let info = { device: "wasm", dtype: "q8" };',
      'async function inferir(audio, language) {',
      '  const opts = { task: "transcribe", return_timestamps: true, chunk_length_s: 20, stride_length_s: 2, no_speech_threshold: 0.5, logprob_threshold: -0.8, compression_ratio_threshold: 2.4, condition_on_previous_text: false };',
      '  if (language) opts.language = language;',
      '  const sr = 16000, win = 30 * sr, hop = 28 * sr;',
      '  if (!audio || audio.length <= win * 1.15) return pipe(audio, opts);',
      '  const chunks = [];',
      '  for (let start = 0; start < audio.length; start += hop) {',
      '    const end = Math.min(audio.length, start + win);',
      '    const piece = audio.slice(start, end);',
      '    const r = await pipe(piece, opts);',
      '    const off = start / sr;',
      '    chunks.push({ r, off });',
      '    if (end >= audio.length) break;',
      '  }',
      '  const segs = [];',
      '  chunks.forEach(({ r, off }) => {',
      '    const list = (r && r.chunks) || [];',
      '    list.forEach(ch => {',
      '      const ts = ch.timestamp || [0, 0];',
      '      segs.push({ timestamp: [(+ts[0] || 0) + off, (ts[1] == null ? (+ts[0] || 0) + 1 : +ts[1]) + off], text: ch.text || "" });',
      '    });',
      '    if (!list.length && r && r.text) segs.push({ timestamp: [off, off + 1], text: r.text });',
      '  });',
      '  return { chunks: segs, text: segs.map(s => s.text).join(" ") };',
      '}',
      'self.onmessage = async (ev) => {',
      '  const { id, tipo, payload } = ev.data || {};',
      '  try {',
      '    if (tipo === "cargar") {',
      '      if (pipe) { self.postMessage({ id, tipo: "ok", info }); return; }',
      '      let device = payload.webgpu ? "webgpu" : "wasm";',
      '      let dtype = payload.webgpu ? "fp16" : "q8";',
      '      try {',
      '        pipe = await pipeline("automatic-speech-recognition", payload.model, {',
      '          device, dtype,',
      '          progress_callback: (p) => self.postMessage({ id, tipo: "progreso", p })',
      '        });',
      '      } catch (e) {',
      '        if (device !== "webgpu") throw e;',
      '        device = "wasm"; dtype = "q8";',
      '        pipe = await pipeline("automatic-speech-recognition", payload.model, {',
      '          device, dtype,',
      '          progress_callback: (p) => self.postMessage({ id, tipo: "progreso", p })',
      '        });',
      '      }',
      '      info = { device, dtype };',
      '      self.postMessage({ id, tipo: "ok", info });',
      '    } else if (tipo === "transcribir") {',
      '      const r = await inferir(payload.audio, payload.language);',
      '      self.postMessage({ id, tipo: "ok", result: r });',
      '    } else if (tipo === "liberar") {',
      '      pipe = null;',
      '      self.postMessage({ id, tipo: "ok" });',
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
      if (d.tipo === 'ok') pend.ok(d.result != null ? d.result : d.info || true);
      else pend.mal(new Error(d.msg || 'error Whisper'));
    };
    worker.onerror = function (e) {
      setEstado('error', e.message || 'worker');
    };
  }

  async function cargarEnHilo(opts) {
    tfMod = await import(TF_URL);
    aplicarEnv(tfMod.env);
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
        deviceUsado = 'wasm';
        dtypeUsado = 'q8';
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
    var want = String(opciones.modelo || TAMANO || 'tiny').toLowerCase();
    if (want !== 'base') want = 'tiny';
    if (want !== TAMANO || MODELO !== MODELOS[want]) {
      try { await liberar(); } catch (_) {}
      TAMANO = want;
      MODELO = MODELOS[want];
    }
    if (estado === 'listo' && (pipe || worker)) {
      return { ok: true, cache: true, device: deviceUsado, dtype: dtypeUsado, modelo: TAMANO };
    }
    if (cargaEnCurso) return cargaEnCurso;
    setEstado('descargando');
    cargaEnCurso = (async function () {
      try {
        var usarWorker = opciones.worker !== false;
        if (usarWorker) {
          try {
            arrancarWorker();
            var info = await llamarWorker('cargar', { model: MODELO, webgpu: webgpu });
            if (info && info.device) {
              deviceUsado = info.device;
              dtypeUsado = info.dtype || dtypeUsado;
            }
            setEstado('listo');
            await marcarOPFS(true);
            return { ok: true, via: 'worker', device: deviceUsado, dtype: dtypeUsado };
          } catch (e) {
            try { worker.terminate(); } catch (_) {}
            worker = null;
          }
        }
        await cargarEnHilo(opciones);
        deviceUsado = webgpu && pipe ? deviceUsado : 'wasm';
        dtypeUsado = deviceUsado === 'webgpu' ? 'fp16' : 'q8';
        setEstado('listo');
        await marcarOPFS(true);
        return { ok: true, via: 'main', device: deviceUsado, dtype: dtypeUsado };
      } catch (e) {
        setEstado('error', e.message || e);
        throw e;
      } finally {
        cargaEnCurso = null;
      }
    })();
    return cargaEnCurso;
  }

  async function transcribir(blob, opciones) {
    opciones = opciones || {};
    var t0 = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    if (estado !== 'listo') {
      if (opciones.exigirListo) throw new Error('Modelo Whisper no listo');
      await cargar({ onProgreso: onProgreso, modelo: opciones.modelo });
    }
    if (!(blob instanceof Blob)) throw new Error('transcribir espera un Blob de audio');
    var audio = await pcm16k(blob);
    var rawLang = String(opciones.idioma || opciones.language || '').toLowerCase();
    var lang = (!rawLang || rawLang === 'auto') ? null : idiomaWhisper(rawLang);
    if (idiomaSesion && lang && idiomaSesion !== lang) idiomaSesion = lang;
    else idiomaSesion = lang || idiomaSesion || '';
    var r;
    if (worker) {
      r = await llamarWorker('transcribir', { audio: audio, language: lang });
    } else if (pipe) {
      var popts = {
        task: 'transcribe',
        return_timestamps: true,
        chunk_length_s: opciones.chunk_length_s || 20,
        stride_length_s: opciones.stride_length_s || 2
      };
      if (lang) popts.language = lang;
      popts.no_speech_threshold = opciones.no_speech_threshold != null ? opciones.no_speech_threshold : 0.5;
      popts.logprob_threshold = opciones.logprob_threshold != null ? opciones.logprob_threshold : -0.8;
      popts.compression_ratio_threshold = opciones.compression_ratio_threshold != null ? opciones.compression_ratio_threshold : 2.4;
      popts.condition_on_previous_text = false;
      r = await pipe(audio, popts);
    } else {
      throw new Error('Whisper no está listo');
    }
    var ms = ((typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now()) - t0;
    return { segmentos: segsDeResultado(r), bruto: r, ms: ms, device: deviceUsado, dtype: dtypeUsado };
  }

  async function liberar() {
    try { if (worker) await llamarWorker('liberar', {}); } catch (_) {}
    try { if (worker) worker.terminate(); } catch (_) {}
    worker = null;
    pipe = null;
    tfMod = null;
    idiomaSesion = '';
    if (estado === 'listo') setEstado('no-descargado');
    return { ok: true };
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
    marcarIdioma: marcarIdioma,
    idiomaListo: idiomaListo,
    borrarIdioma: borrarIdioma,
    listarIdiomas: leerLangs,
    bytesCache: bytesCache,
    estado: function () { return estado; },
    error: function () { return ultimoError; },
    liberar: liberar,
    webgpu: function () { return webgpuOk; },
    device: function () { return deviceUsado; },
    dtype: function () { return dtypeUsado; },
    etiquetaModelo: function () {
      var q = dtypeUsado === 'q8' ? 'int8' : (dtypeUsado || 'fp32');
      return TAMANO + '-' + q;
    },
    modelo: function () { return MODELO; },
    tamano: function () { return TAMANO; },
    vad: vadSilero,
    asegurarVad: asegurarSilero,
    sileroListo: function () { return !!sileroListo && !!sesSilero; },
    hayCache: async function () { return (await hayMarcaOPFS()) || (await hayCacheHF()); }
  };
})(typeof window !== 'undefined' ? window : self);
