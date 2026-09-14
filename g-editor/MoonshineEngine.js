/* MoonshineEngine.js · ASR en vivo (PC). Audio 100% local.
   API: cargar / transcribir / estado / borrar
   Modelo: onnx-community/moonshine-tiny-ONNX vía transformers.js
*/
(function (global) {
  'use strict';

  var MODELO = 'onnx-community/moonshine-tiny-ONNX';
  var TF_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2';
  var TF_DIST = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2/dist/';
  var HF_HOST = 'https://huggingface.co/';

  var estado = 'no-descargado';
  var ultimoError = '';
  var pipe = null;
  var tfMod = null;
  var onProgreso = null;
  var cargaEnCurso = null;
  var webgpu = typeof navigator !== 'undefined' && !!navigator.gpu;
  var webgpuOk = false;

  function log() {
    try { console.info.apply(console, ['[Moonshine]'].concat([].slice.call(arguments))); } catch (_) {}
  }

  function setEstado(s, err) {
    estado = s;
    if (err) ultimoError = String(err);
    try { global.dispatchEvent(new CustomEvent('moonshine-estado', { detail: estado })); } catch (_) {}
  }

  function aplicarEnv(env) {
    if (!env) return;
    env.allowLocalModels = false;
    env.allowRemoteModels = true;
    env.useBrowserCache = true;
    env.remoteHost = HF_HOST;
    env.remotePathTemplate = '{model}/resolve/{revision}/';
    try {
      if (env.backends && env.backends.onnx && env.backends.onnx.wasm)
        env.backends.onnx.wasm.wasmPaths = TF_DIST;
        var aislado = typeof crossOriginIsolated !== 'undefined' && !!crossOriginIsolated;
        var cores = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 1;
        env.backends.onnx.wasm.numThreads = aislado ? Math.min(4, cores) : 1;
        env.backends.onnx.wasm.simd = true;
        log('WASM threads', env.backends.onnx.wasm.numThreads, 'COOP/COEP', aislado);
    } catch (e) { log('env wasmPaths', e && e.message); }
  }

  async function dirMotor(crear) {
    if (!navigator.storage || !navigator.storage.getDirectory) return null;
    var root = await navigator.storage.getDirectory();
    return root.getDirectoryHandle('moonshine-engine', { create: !!crear });
  }

  async function marcarOPFS(listo) {
    try {
      if (!listo) {
        var root = await navigator.storage.getDirectory();
        try { await root.removeEntry('moonshine-engine', { recursive: true }); } catch (_) {}
        return;
      }
      var d = await dirMotor(true);
      if (!d) return;
      var f = await d.getFileHandle('ready.txt', { create: true });
      var w = await f.createWritable();
      await w.write(MODELO + '\n' + Date.now());
      await w.close();
    } catch (e) { log('OPFS', e && e.message); }
  }

  async function pcm16k(blob) {
    var AC = global.AudioContext || global.webkitAudioContext;
    var ac = new AC();
    var raw = await blob.arrayBuffer();
    var buf;
    try { buf = await ac.decodeAudioData(raw.slice(0)); }
    finally { try { ac.close(); } catch (_) {} }
    var n = buf.length, chs = buf.numberOfChannels;
    var mono = new Float32Array(n);
    for (var c = 0; c < chs; c++) {
      var ch = buf.getChannelData(c);
      for (var i = 0; i < n; i++) mono[i] += ch[i] / chs;
    }
    var sr = buf.sampleRate;
    if (sr === 16000) return { pcm: mono, dur: n / sr };
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
    return { pcm: out, dur: outLen / 16000 };
  }

  async function cargar(opciones) {
    opciones = opciones || {};
    onProgreso = opciones.onProgreso || opciones.onProgress || null;
    if (estado === 'listo' && pipe) return { ok: true, cache: true };
    if (cargaEnCurso) return cargaEnCurso;
    setEstado('descargando');
    cargaEnCurso = (async function () {
      try {
        log('cargando', MODELO);
        tfMod = await import(TF_URL);
        aplicarEnv(tfMod.env);
        webgpuOk = false;
        if (webgpu && navigator.gpu) {
          try {
            var ad = await navigator.gpu.requestAdapter();
            webgpuOk = !!ad;
            log('INFO adapter', ad ? 'OK' : 'null');
          } catch (e) { log('INFO probe WebGPU', e && e.message); }
        }
        var device = webgpuOk ? 'webgpu' : 'wasm';
        var dtype = webgpuOk ? 'fp16' : 'q4';
        try {
          pipe = await tfMod.pipeline('automatic-speech-recognition', MODELO, {
            device: device,
            dtype: dtype,
            progress_callback: function (p) {
              if (onProgreso) onProgreso({
                stage: p.status || '',
                file: p.file || '',
                progress: typeof p.progress === 'number' ? p.progress : 0
              });
            }
          });
        } catch (e) {
          log('INFO WebGPU/dtype falló, WASM q8', e && e.message);
          pipe = await tfMod.pipeline('automatic-speech-recognition', MODELO, {
            device: 'wasm',
            dtype: 'q8',
            progress_callback: function (p) {
              if (onProgreso) onProgreso({
                stage: p.status || '',
                file: p.file || '',
                progress: typeof p.progress === 'number' ? p.progress : 0
              });
            }
          });
        }
        setEstado('listo');
        await marcarOPFS(true);
        log('modelo cargado');
        return { ok: true };
      } catch (e) {
        setEstado('error', e && e.message || e);
        log('ERROR carga', e && e.message || e);
        throw e;
      } finally {
        cargaEnCurso = null;
      }
    })();
    return cargaEnCurso;
  }

  async function transcribir(blob) {
    if (estado !== 'listo') await cargar({ onProgreso: onProgreso });
    if (!(blob instanceof Blob)) throw new Error('transcribir espera un Blob');
    var t0 = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    var dec = await pcm16k(blob);
    var r = await pipe(dec.pcm);
    var ms = ((typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now()) - t0;
    log('chunk procesado en', Math.round(ms), 'ms');
    var segs = [];
    if (r && Array.isArray(r.chunks) && r.chunks.length) {
      r.chunks.forEach(function (ch) {
        var ts = ch.timestamp || [0, dec.dur];
        var texto = String(ch.text || '').trim();
        if (texto) segs.push({ t0: +ts[0] || 0, t1: ts[1] == null ? dec.dur : +ts[1], texto: texto });
      });
    } else {
      var txt = String((r && r.text) || '').trim();
      if (txt) segs.push({ t0: 0, t1: dec.dur || 1, texto: txt });
    }
    return { segmentos: segs, bruto: r, ms: ms };
  }

  async function borrar() {
    pipe = null;
    tfMod = null;
    try {
      var ks = await caches.keys();
      for (var i = 0; i < ks.length; i++) {
        if (/moonshine|huggingface|transformers/i.test(ks[i])) await caches.delete(ks[i]);
      }
    } catch (e) { log('ERROR borrar cache', e && e.message); }
    await marcarOPFS(false);
    setEstado('no-descargado');
    log('modelo borrado');
    return { ok: true };
  }

  global.MoonshineEngine = {
    cargar: cargar,
    transcribir: transcribir,
    borrar: borrar,
    estado: function () { return estado; },
    error: function () { return ultimoError; },
    modelo: function () { return MODELO; }
  };
})(typeof window !== 'undefined' ? window : self);
