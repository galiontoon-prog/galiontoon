/* DiarizationEngine.js · motor OPCIONAL. Misma carpeta que el HTML.
   API: cargar / diarizar / fusionar / estado / borrar / hayCache

   Profesional: pyannote-segmentation-3.0 vía transformers.js
   (ONNX, WebGPU → WASM). Cache: Cache API + marca OPFS.

   GitHub Releases: si GH_RELEASE apunta a tus .onnx, se bajan
   por CORS_PROXY y se guardan en OPFS. Si falla, HF.

   Ligero: embeddings de timbre si el modelo no está.
*/
(function (global) {
  'use strict';

  var MODELO_HF = 'onnx-community/pyannote-segmentation-3.0';
  var TF_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2/+esm';
  var TF_DIST = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2/dist/';
  var HF_HOST = 'https://huggingface.co/';
  var GH_RELEASE = '';
  var CORS_PROXY = '';
  var ARCHIVOS_GH = ['segmentation.onnx', 'config.json'];
  var TAM_MB = 6;

  var estado = 'no-descargado';
  var ultimoError = '';
  var tfMod = null;
  var modelo = null;
  var processor = null;
  var webgpu = typeof navigator !== 'undefined' && !!navigator.gpu;
  var onProgreso = null;
  var cargaEnCurso = null;

  function postEstado() {
    try { global.dispatchEvent(new CustomEvent('diar-estado', { detail: estado })); } catch (_) {}
  }
  function setEstado(s, err) {
    estado = s;
    if (err) ultimoError = String(err);
    postEstado();
  }

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

  async function dirMotor(crear) {
    if (!navigator.storage || !navigator.storage.getDirectory) return null;
    var root = await navigator.storage.getDirectory();
    return root.getDirectoryHandle('diar-engine', { create: !!crear });
  }

  async function marcarOPFS(listo) {
    try {
      if (!listo) {
        if (!navigator.storage || !navigator.storage.getDirectory) return;
        var root = await navigator.storage.getDirectory();
        try { await root.removeEntry('diar-engine', { recursive: true }); } catch (_) {}
        return;
      }
      var d = await dirMotor(true);
      if (!d) return;
      var f = await d.getFileHandle('ready.txt', { create: true });
      var w = await f.createWritable();
      await w.write(MODELO_HF + '\n' + Date.now());
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

  async function hayCacheHF() {
    try {
      var ks = await caches.keys();
      return ks.some(function (k) { return /huggingface|transformers|pyannote|diar/i.test(k); });
    } catch (_) { return false; }
  }

  function urlViaProxy(url) {
    if (!CORS_PROXY) return url;
    return CORS_PROXY + encodeURIComponent(url);
  }

  async function bajarGH(onProg) {
    if (!GH_RELEASE) return false;
    var d = await dirMotor(true);
    if (!d) return false;
    var ok = 0;
    for (var i = 0; i < ARCHIVOS_GH.length; i++) {
      var nom = ARCHIVOS_GH[i];
      var url = urlViaProxy(GH_RELEASE.replace(/\/?$/, '/') + nom);
      try {
        var res = await fetch(url);
        if (!res.ok) continue;
        var buf = await res.arrayBuffer();
        var f = await d.getFileHandle(nom, { create: true });
        var w = await f.createWritable();
        await w.write(buf);
        await w.close();
        ok++;
        if (onProg) onProg({ progress: ((i + 1) / ARCHIVOS_GH.length) * 100, file: nom });
      } catch (_) {}
    }
    return ok > 0;
  }

  async function pcm16k(blob) {
    var AC = global.AudioContext || global.webkitAudioContext;
    if (!AC || !blob) return null;
    var ac = new AC();
    var raw;
    try { raw = await blob.arrayBuffer(); } catch (e) { try { ac.close(); } catch (_) {} throw e; }
    var buf;
    try { buf = await ac.decodeAudioData(raw.slice(0)); }
    finally { try { ac.close(); } catch (_) {} }
    var n = buf.length, chs = buf.numberOfChannels, sr = buf.sampleRate;
    var mono = new Float32Array(n);
    for (var c = 0; c < chs; c++) {
      var ch = buf.getChannelData(c);
      for (var i = 0; i < n; i++) mono[i] += ch[i] / chs;
    }
    if (sr === 16000) return mono;
    var ratio = sr / 16000, outLen = Math.max(1, Math.round(n / ratio));
    var out = new Float32Array(outLen);
    for (var j = 0; j < outLen; j++) {
      var x = j * ratio, i0 = Math.floor(x), i1 = Math.min(n - 1, i0 + 1), f = x - i0;
      out[j] = mono[i0] * (1 - f) + mono[i1] * f;
    }
    return out;
  }

  function argmaxFila(arr, off, dim) {
    var best = 0, val = arr[off];
    for (var i = 1; i < dim; i++) if (arr[off + i] > val) { val = arr[off + i]; best = i; }
    return best;
  }

  function labelAHablante(lab) {
    var s = String(lab == null ? '' : lab).toLowerCase();
    if (!s || /non[-_]?speech|silence|none|empty/.test(s)) return '';
    var m = s.match(/speaker[_ ]?(\d+)/);
    if (m) return String((+m[1]) + 1);
    if (/^\d+$/.test(s)) {
      var n = +s;
      return n === 0 ? '' : String(n);
    }
    return s ? '1' : '';
  }

  function framesATurnos(clases, labels, dur, hop) {
    var turnos = [];
    var cur = '', t0 = 0;
    var n = clases.length;
    hop = hop || (dur / Math.max(1, n));
    for (var i = 0; i <= n; i++) {
      var hab = i < n ? labelAHablante(labels && labels[clases[i]] != null ? labels[clases[i]] : clases[i]) : '';
      if (hab !== cur) {
        if (cur) turnos.push({ t0: t0, t1: Math.min(dur, i * hop), hablante: cur });
        cur = hab;
        t0 = i * hop;
      }
    }
    return turnos.filter(function (t) { return t.t1 - t.t0 >= 0.12; });
  }

  function solape(a0, a1, b0, b1) {
    return Math.max(0, Math.min(a1, b1) - Math.max(a0, b0));
  }

  function feats(pcm, sr, t0, t1) {
    var a = Math.max(0, Math.floor(t0 * sr));
    var b = Math.min(pcm.length, Math.floor(t1 * sr));
    if (b - a < sr * 0.12) return null;
    var sl = pcm.subarray(a, b);
    var e = 0, zc = 0, i;
    for (i = 0; i < sl.length; i++) {
      e += sl[i] * sl[i];
      if (i && ((sl[i] >= 0) !== (sl[i - 1] >= 0))) zc++;
    }
    var rms = Math.sqrt(e / sl.length);
    if (rms < 0.004) return null;
    return [Math.log(rms + 1e-6), zc / sl.length, rms];
  }

  function clusterLigero(vecs, maxK) {
    var n = vecs.length;
    if (!n) return [];
    var lab = vecs.map(function (_, i) { return i; });
    function dist(i, j) {
      var s = 0;
      for (var d = 0; d < vecs[i].length; d++) {
        var x = vecs[i][d] - vecs[j][d]; s += x * x;
      }
      return Math.sqrt(s);
    }
    var TH = 0.8;
    while (new Set(lab).size > 1) {
      var ids = Array.from(new Set(lab)), best = Infinity, ia = -1, ib = -1, a, b, p, q;
      for (a = 0; a < ids.length; a++) for (b = a + 1; b < ids.length; b++) {
        var acc = 0, cnt = 0;
        for (p = 0; p < n; p++) if (lab[p] === ids[a])
          for (q = 0; q < n; q++) if (lab[q] === ids[b]) { acc += dist(p, q); cnt++; }
        var dd = cnt ? acc / cnt : Infinity;
        if (dd < best) { best = dd; ia = ids[a]; ib = ids[b]; }
      }
      if (ia < 0 || (new Set(lab).size <= maxK && best > TH) || best > TH * 2) break;
      for (p = 0; p < n; p++) if (lab[p] === ib) lab[p] = ia;
    }
    var map = {}, out = [], k = 0;
    for (var i = 0; i < n; i++) {
      if (map[lab[i]] == null) map[lab[i]] = String(++k);
      out.push(map[lab[i]]);
    }
    return out;
  }

  async function cargar(opciones) {
    opciones = opciones || {};
    onProgreso = opciones.onProgreso || opciones.onProgress || null;
    if (modelo && processor) { setEstado('listo'); return { ok: true, cache: true, modo: 'profesional' }; }
    if (cargaEnCurso) return cargaEnCurso;
    setEstado('descargando');
    cargaEnCurso = (async function () {
      try {
        if (GH_RELEASE) {
          try { await bajarGH(onProgreso); } catch (_) {}
        }
        tfMod = await import(TF_URL);
        aplicarEnv(tfMod.env);
        var device = webgpu ? 'webgpu' : 'wasm';
        var opts = {
          device: device,
          dtype: webgpu ? 'fp16' : 'q8',
          progress_callback: function (p) {
            if (onProgreso) {
              var pct = typeof p.progress === 'number' ? p.progress : 0;
              onProgreso({ stage: p.status || '', file: p.file || '', progress: pct <= 1 ? pct * 100 : pct });
            }
          }
        };
        try {
          processor = await tfMod.AutoProcessor.from_pretrained(MODELO_HF, opts);
          modelo = await tfMod.AutoModelForAudioFrameClassification.from_pretrained(MODELO_HF, opts);
        } catch (e) {
          if (device !== 'webgpu') throw e;
          opts.device = 'wasm'; opts.dtype = 'q8';
          processor = await tfMod.AutoProcessor.from_pretrained(MODELO_HF, opts);
          modelo = await tfMod.AutoModelForAudioFrameClassification.from_pretrained(MODELO_HF, opts);
        }
        await marcarOPFS(true);
        setEstado('listo');
        return { ok: true, modo: 'profesional', modelo: MODELO_HF, webgpu: webgpu };
      } catch (e) {
        setEstado('error', e.message || e);
        throw e;
      } finally {
        cargaEnCurso = null;
      }
    })();
    return cargaEnCurso;
  }

  async function diarizarProfesional(audio) {
    var inputs = await processor(audio);
    var out = await modelo(inputs);
    var logits = out && (out.logits || out);
    if (!logits) throw new Error('Sin logits de pyannote');
    if (processor.post_process_speaker_diarization) {
      var segs = processor.post_process_speaker_diarization(logits, audio.length);
      var lista = (segs && segs[0]) || segs || [];
      return lista.map(function (s) {
        var id = (s.id == null ? 0 : +s.id);
        return { t0: +s.start || 0, t1: +s.end || 0, hablante: String(id + 1) };
      }).filter(function (t) { return t.t1 - t.t0 >= 0.12; });
    }
    var data = logits.data || logits;
    var dims = logits.dims || [];
    var frames, clases;
    if (dims.length === 3) { frames = dims[1]; clases = dims[2]; }
    else if (dims.length === 2) { frames = dims[0]; clases = dims[1]; }
    else {
      var id2 = (modelo.config && modelo.config.id2label) || {};
      clases = Object.keys(id2).length || 7;
      frames = Math.floor(data.length / clases);
    }
    var ids = [];
    for (var f = 0; f < frames; f++) ids.push(argmaxFila(data, f * clases, clases));
    var labels = (modelo.config && modelo.config.id2label) || null;
    var dur = audio.length / 16000;
    var hop = dur / Math.max(1, frames);
    return framesATurnos(ids, labels, dur, hop);
  }

  async function diarizarLigero(blob, cues, maxK) {
    var wav = await pcm16k(blob);
    if (!wav) throw new Error('No se pudo leer el audio');
    var sr = 16000;
    var turnos = [];
    if (cues && cues.length) {
      var vecs = [], idx = [];
      cues.forEach(function (c, i) {
        var f = feats(wav, sr, c.t0, c.t1);
        if (f) { vecs.push(f); idx.push(i); }
      });
      var labs = clusterLigero(vecs, maxK);
      var asig = new Array(cues.length).fill('');
      idx.forEach(function (i, k) { asig[i] = labs[k] || '1'; });
      for (var i = 0; i < asig.length; i++) if (!asig[i]) asig[i] = (i && asig[i - 1]) ? asig[i - 1] : '1';
      cues.forEach(function (c, ix) {
        turnos.push({ t0: c.t0, t1: c.t1, hablante: asig[ix] || '1' });
      });
    }
    return turnos;
  }

  async function diarizar(blob, opciones) {
    opciones = opciones || {};
    var maxK = Math.max(2, Math.min(5, +opciones.maxK || 3));
    var cues = opciones.cues || [];
    var modo = opciones.modo || 'auto';
    if (modo === 'ligero' || modo === 'rapido') {
      return { turnos: await diarizarLigero(blob, cues, maxK), modo: 'ligero' };
    }
    if (estado !== 'listo' || !modelo) {
      try { await cargar({ onProgreso: onProgreso }); }
      catch (e) {
        if (modo === 'profesional' && opciones.fallback === false) throw e;
        return { turnos: await diarizarLigero(blob, cues, maxK), modo: 'ligero', aviso: e.message || 'fallback' };
      }
    }
    try {
      var audio = await pcm16k(blob);
      if (!audio) throw new Error('audio');
      var turnos = await diarizarProfesional(audio);
      if (!turnos.length && cues.length) {
        return { turnos: await diarizarLigero(blob, cues, maxK), modo: 'ligero', aviso: 'sin-turnos' };
      }
      return { turnos: turnos, modo: 'profesional', modelo: MODELO_HF };
    } catch (e) {
      if (modo === 'profesional' && opciones.fallback === false) throw e;
      return { turnos: await diarizarLigero(blob, cues, maxK), modo: 'ligero', aviso: e.message || 'fallback' };
    }
  }

  function fusionar(cues, turnos) {
    return (cues || []).map(function (c) {
      var best = '', bestS = 0;
      (turnos || []).forEach(function (t) {
        var s = solape(c.t0, c.t1, t.t0, t.t1);
        if (s > bestS) { bestS = s; best = t.hablante; }
      });
      return { id: c.id, t0: c.t0, t1: c.t1, hablante: best || '1' };
    });
  }

  async function borrar() {
    modelo = null; processor = null; tfMod = null;
    try {
      var ks = await caches.keys();
      for (var i = 0; i < ks.length; i++) {
        if (/huggingface|transformers|pyannote|diar/i.test(ks[i])) await caches.delete(ks[i]);
      }
    } catch (_) {}
    await marcarOPFS(false);
    setEstado('no-descargado');
    return { ok: true };
  }

  (async function probe() {
    try {
      if (await hayMarcaOPFS() || await hayCacheHF()) setEstado('no-descargado');
    } catch (_) {}
  })();

  global.DiarizationEngine = {
    cargar: cargar,
    diarizar: diarizar,
    fusionar: fusionar,
    borrar: borrar,
    estado: function () { return estado; },
    error: function () { return ultimoError; },
    modelo: function () { return MODELO_HF; },
    mb: function () { return TAM_MB; },
    hayCache: async function () { return (await hayMarcaOPFS()) || (await hayCacheHF()); }
  };
})(typeof window !== 'undefined' ? window : self);
