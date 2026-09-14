/* DiarizationEngine.js · motor OPCIONAL. Misma carpeta que el HTML.
   API: cargar / diarizar / fusionar / estado / borrar / hayCache

   Profesional (sin Hugging Face, sin token):
     fp16 (~28 MB) o fp32 (~31 MB)
     GitHub Releases vía Cloudflare Worker (CORS).

   Ligero: timbre local si falta red, ONNX o archivos.
*/
(function (global) {
  'use strict';

  var CORS_PROXY = 'https://galiontoon-cors.galiontoon.workers.dev/?url=';
  var RELEASE_BASE = 'https://github.com/galiontoon-prog/galiontoon/releases/download/modelos-diarizacion-v1/';
  var VARIANTES = {
    fp16: { id: 'fp16', nom: 'Rápido', mb: 28, seg: 'segmentacion-fp16.onnx', emb: 'embeddings.onnx' },
    fp32: { id: 'fp32', nom: 'Máxima', mb: 31, seg: 'segmentacion-fp32.onnx', emb: 'embeddings.onnx' }
  };
  var EXTRA = ['config-segmentacion.json', 'preprocessor-segmentacion.json', 'config-embeddings.yaml'];
  var ORT_SRC = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js';
  var ORT_WASM = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

  var estado = 'no-descargado';
  var ultimoError = '';
  var onProgreso = null;
  var cargaEnCurso = null;
  var sesSeg = null;
  var sesEmb = null;
  var sesiones = { fp16: { seg: null, emb: null }, fp32: { seg: null, emb: null } };
  var activa = 'fp16';
  var webgpu = typeof navigator !== 'undefined' && !!navigator.gpu;
  function pack(v) { return VARIANTES[v] || VARIANTES.fp16; }

  function serverBase() { return RELEASE_BASE; }
  function urlModelo(nom) {
    return CORS_PROXY + RELEASE_BASE + nom;
  }

  function postEstado() {
    try { global.dispatchEvent(new CustomEvent('diar-estado', { detail: estado })); } catch (_) {}
  }
  function setEstado(s, err) {
    estado = s;
    if (err) ultimoError = String(err);
    postEstado();
  }

  async function dirMotor(crear) {
    if (!navigator.storage || !navigator.storage.getDirectory) return null;
    var root = await navigator.storage.getDirectory();
    return root.getDirectoryHandle('diar-engine', { create: !!crear });
  }

  async function leerOPFS(nom) {
    try {
      var d = await dirMotor(false);
      if (!d) return null;
      var f = await d.getFileHandle(nom, { create: false });
      var file = await f.getFile();
      if (!file || file.size < 1024) return null;
      return await file.arrayBuffer();
    } catch (_) { return null; }
  }

  async function escribirOPFS(nom, buf) {
    var d = await dirMotor(true);
    if (!d) return false;
    var f = await d.getFileHandle(nom, { create: true });
    var w = await f.createWritable();
    await w.write(buf);
    await w.close();
    return true;
  }

  async function hayCache(v) {
    var p = pack(v || activa);
    var a = await leerOPFS(p.seg);
    var b = await leerOPFS(p.emb);
    return !!(a && b);
  }
  async function borrarArchivo(nom) {
    try {
      var d = await dirMotor(false);
      if (!d) return;
      await d.removeEntry(nom);
    } catch (_) {}
  }

  async function bajarConProgreso(url, pesoRel, pesoBase) {
    var res = await fetch(url, { mode: 'cors' });
    if (!res.ok) throw new Error('HTTP ' + res.status + ' ' + url);
    var total = +res.headers.get('content-length') || 0;
    if (!res.body || !res.body.getReader) {
      var all = await res.arrayBuffer();
      if (onProgreso) onProgreso({ file: url, progress: pesoBase + pesoRel });
      return all;
    }
    var reader = res.body.getReader();
    var chunks = [];
    var rec = 0;
    while (true) {
      var step = await reader.read();
      if (step.done) break;
      chunks.push(step.value);
      rec += step.value.length;
      if (onProgreso) {
        var frac = total ? rec / total : 0.5;
        onProgreso({ file: url, progress: pesoBase + frac * pesoRel });
      }
    }
    var out = new Uint8Array(rec);
    var o = 0;
    for (var i = 0; i < chunks.length; i++) { out.set(chunks[i], o); o += chunks[i].length; }
    return out.buffer;
  }

  async function asegurarOrt() {
    if (global.ort) return global.ort;
    await new Promise(function (ok, bad) {
      var s = document.createElement('script');
      s.src = ORT_SRC;
      s.onload = ok;
      s.onerror = function () { bad(new Error('No cargó ONNX Runtime')); };
      document.head.appendChild(s);
    });
    try {
      var w = global.ort.env.wasm;
      w.wasmPaths = ORT_WASM;
      w.simd = true;
      var aislado = typeof crossOriginIsolated !== 'undefined' && !!crossOriginIsolated;
      var cores = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 1;
      w.numThreads = aislado ? Math.min(4, cores) : 1;
      w.proxy = !!aislado;
    } catch (_) {}
    return global.ort;
  }

  async function crearSesion(buf) {
    var ort = await asegurarOrt();
    var gpu = false;
    if (webgpu && navigator.gpu) {
      try { gpu = !!(await navigator.gpu.requestAdapter()); } catch (_) { gpu = false; }
    }
    var providers = gpu ? ['webgpu', 'wasm'] : ['wasm'];
    var last;
    for (var i = 0; i < providers.length; i++) {
      try {
        return await ort.InferenceSession.create(buf, { executionProviders: [providers[i]] });
      } catch (e) { last = e; }
    }
    throw last || new Error('ONNX session');
  }

  async function cargar(opciones) {
    opciones = opciones || {};
    onProgreso = opciones.onProgreso || opciones.onProgress || null;
    var v = opciones.variante || activa || 'fp16';
    activa = v;
    var p = pack(v);
    if (sesiones[v] && sesiones[v].seg && sesiones[v].emb) {
      sesSeg = sesiones[v].seg; sesEmb = sesiones[v].emb;
      setEstado('listo');
      return { ok: true, cache: true, modo: 'profesional', variante: v };
    }
    if (cargaEnCurso) return cargaEnCurso;
    setEstado('descargando');
    cargaEnCurso = (async function () {
      try {
        var segBuf = await leerOPFS(p.seg);
        var embBuf = await leerOPFS(p.emb);
        if (!segBuf) {
          if (onProgreso) onProgreso({ file: p.seg, progress: 2 });
          segBuf = await bajarConProgreso(urlModelo(p.seg), 30, 5);
          await escribirOPFS(p.seg, segBuf);
        }
        if (!embBuf) {
          if (onProgreso) onProgreso({ file: p.emb, progress: 40 });
          embBuf = await bajarConProgreso(urlModelo(p.emb), 50, 40);
          await escribirOPFS(p.emb, embBuf);
        }
        for (var ei = 0; ei < EXTRA.length; ei++) {
          try {
            if (!(await leerOPFS(EXTRA[ei]))) {
              var extraBuf = await bajarConProgreso(urlModelo(EXTRA[ei]), 1, 91);
              await escribirOPFS(EXTRA[ei], extraBuf);
            }
          } catch (_) {}
        }
        if (onProgreso) onProgreso({ file: 'sesión', progress: 96 });
        sesSeg = await crearSesion(segBuf);
        sesEmb = await crearSesion(embBuf);
        sesiones[v] = { seg: sesSeg, emb: sesEmb };
        setEstado('listo');
        return { ok: true, modo: 'profesional', variante: v, origen: urlModelo(''), webgpu: webgpu, mb: p.mb };
      } catch (e) {
        setEstado('error', e.message || e);
        throw e;
      } finally {
        cargaEnCurso = null;
      }
    })();
    return cargaEnCurso;
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

  function tensorDe(ort, arr, dims) {
    return new ort.Tensor('float32', arr, dims);
  }

  function pickInput(ses, prefer) {
    var names = ses.inputNames || [];
    for (var i = 0; i < prefer.length; i++) {
      if (names.indexOf(prefer[i]) >= 0) return prefer[i];
    }
    return names[0];
  }

  async function correrSeg(pcm) {
    if (!sesSeg) throw new Error('sin segmentación');
    var ort = global.ort;
    var name = pickInput(sesSeg, ['waveform', 'input', 'audio', 'input_values']);
    var feeds = {};
    feeds[name] = tensorDe(ort, pcm, [1, pcm.length]);
    try {
      return await sesSeg.run(feeds);
    } catch (_) {
      feeds[name] = tensorDe(ort, pcm, [1, 1, pcm.length]);
      return await sesSeg.run(feeds);
    }
  }

  function logitsATurnos(out, dur) {
    var key = (sesSeg.outputNames && sesSeg.outputNames[0]) || Object.keys(out)[0];
    var ten = out[key];
    if (!ten) return [];
    var data = ten.data;
    var dims = ten.dims || [];
    var frames, clases;
    if (dims.length === 3) { frames = dims[1]; clases = dims[2]; }
    else if (dims.length === 2) { frames = dims[0]; clases = dims[1]; }
    else { clases = 7; frames = Math.floor(data.length / clases); }
    var hop = dur / Math.max(1, frames);
    var turnos = [];
    var cur = '', t0 = 0;
    for (var i = 0; i <= frames; i++) {
      var hab = '';
      if (i < frames) {
        var off = i * clases, best = 0, val = data[off];
        for (var c = 1; c < clases; c++) if (data[off + c] > val) { val = data[off + c]; best = c; }
        hab = best === 0 ? '' : String(best);
      }
      if (hab !== cur) {
        if (cur) turnos.push({ t0: t0, t1: Math.min(dur, i * hop), hablante: cur });
        cur = hab;
        t0 = i * hop;
      }
    }
    return turnos.filter(function (t) { return t.t1 - t.t0 >= 0.18; });
  }

  function vadTurnos(pcm, sr) {
    var win = Math.floor(sr * 0.02);
    var th = 0.012, on = false, t0 = 0;
    var out = [];
    for (var i = 0; i + win <= pcm.length; i += win) {
      var e = 0;
      for (var k = 0; k < win; k++) e += pcm[i + k] * pcm[i + k];
      var rms = Math.sqrt(e / win);
      var t = i / sr;
      if (!on && rms >= th) { on = true; t0 = t; }
      if (on && rms < th * 0.6) {
        if (t - t0 >= 0.25) out.push({ t0: t0, t1: t, hablante: '1' });
        on = false;
      }
    }
    if (on && pcm.length / sr - t0 >= 0.25) out.push({ t0: t0, t1: pcm.length / sr, hablante: '1' });
    return out;
  }

  function hann(n) {
    var w = new Float32Array(n);
    for (var i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / Math.max(1, n - 1));
    return w;
  }

  function hzMel(hz) { return 2595 * Math.log10(1 + hz / 700); }
  function melHz(m) { return 700 * (Math.pow(10, m / 2595) - 1); }

  function filtroMel(nfft, sr, nmel) {
    var fmax = sr / 2;
    var mmin = hzMel(0), mmax = hzMel(fmax);
    var pts = [];
    for (var i = 0; i < nmel + 2; i++) pts.push(melHz(mmin + (mmax - mmin) * i / (nmel + 1)));
    var bins = pts.map(function (f) { return Math.floor((nfft + 1) * f / sr); });
    var w = [];
    for (var m = 0; m < nmel; m++) {
      var row = new Float32Array(nfft / 2 + 1);
      var k;
      for (k = bins[m]; k < bins[m + 1]; k++) if (k >= 0 && k < row.length) row[k] = (k - bins[m]) / Math.max(1, bins[m + 1] - bins[m]);
      for (k = bins[m + 1]; k < bins[m + 2]; k++) if (k >= 0 && k < row.length) row[k] = (bins[m + 2] - k) / Math.max(1, bins[m + 2] - bins[m + 1]);
      w.push(row);
    }
    return w;
  }

  function dftMag(frame) {
    var n = frame.length, half = n / 2 + 1;
    var mag = new Float32Array(half);
    for (var k = 0; k < half; k++) {
      var re = 0, im = 0;
      for (var t = 0; t < n; t++) {
        var ang = (2 * Math.PI * k * t) / n;
        re += frame[t] * Math.cos(ang);
        im -= frame[t] * Math.sin(ang);
      }
      mag[k] = re * re + im * im;
    }
    return mag;
  }

  function fbank80(pcm, sr) {
    var nfft = 512, winN = Math.floor(sr * 0.025), hop = Math.floor(sr * 0.01);
    var w = hann(winN);
    var mels = filtroMel(nfft, sr, 80);
    var nfr = Math.max(1, Math.floor((pcm.length - winN) / hop) + 1);
    var out = new Float32Array(80 * nfr);
    var frame = new Float32Array(nfft);
    var means = new Float32Array(80);
    var i, m, t;
    for (i = 0; i < nfr; i++) {
      frame.fill(0);
      var a = i * hop;
      for (t = 0; t < winN && a + t < pcm.length; t++) frame[t] = pcm[a + t] * w[t];
      var mag = dftMag(frame);
      for (m = 0; m < 80; m++) {
        var s = 0;
        var row = mels[m];
        for (var k = 0; k < row.length; k++) s += mag[k] * row[k];
        var v = Math.log(s + 1e-6);
        out[m * nfr + i] = v;
        means[m] += v;
      }
    }
    for (m = 0; m < 80; m++) means[m] /= nfr;
    for (i = 0; i < nfr; i++) for (m = 0; m < 80; m++) out[m * nfr + i] -= means[m];
    return { data: out, frames: nfr };
  }

  async function embedTurno(pcm, t0, t1) {
    var sr = 16000;
    var a = Math.max(0, Math.floor(t0 * sr));
    var b = Math.min(pcm.length, Math.floor(t1 * sr));
    if (b - a < sr * 0.2) return null;
    var sl = pcm.subarray(a, b);
    var fb = fbank80(sl, sr);
    var ort = global.ort;
    var name = pickInput(sesEmb, ['feats', 'feature', 'input', 'fbank', 'audio']);
    var shapes = [
      [1, 80, fb.frames],
      [1, 1, 80, fb.frames],
      [1, fb.frames, 80]
    ];
    var last;
    for (var i = 0; i < shapes.length; i++) {
      try {
        var feeds = {};
        var arr = fb.data;
        if (shapes[i][1] === fb.frames && shapes[i][2] === 80) {
          arr = new Float32Array(80 * fb.frames);
          for (var tt = 0; tt < fb.frames; tt++)
            for (var mm = 0; mm < 80; mm++) arr[tt * 80 + mm] = fb.data[mm * fb.frames + tt];
        }
        feeds[name] = tensorDe(ort, arr, shapes[i]);
        var out = await sesEmb.run(feeds);
        var key = (sesEmb.outputNames && sesEmb.outputNames[0]) || Object.keys(out)[0];
        var ten = out[key];
        var vec = Array.prototype.slice.call(ten.data);
        var nrm = 0; for (var j = 0; j < vec.length; j++) nrm += vec[j] * vec[j];
        nrm = Math.sqrt(nrm) || 1;
        for (j = 0; j < vec.length; j++) vec[j] /= nrm;
        return vec;
      } catch (e) { last = e; }
    }
    throw last || new Error('embedding');
  }

  function clusterEmb(vecs, maxK) {
    var n = vecs.length;
    if (!n) return [];
    var lab = vecs.map(function (_, i) { return i; });
    function cos(i, j) {
      var s = 0;
      for (var d = 0; d < vecs[i].length; d++) s += vecs[i][d] * vecs[j][d];
      return 1 - s;
    }
    var TH = 0.35;
    while (new Set(lab).size > 1) {
      var ids = Array.from(new Set(lab)), best = Infinity, ia = -1, ib = -1, a, b, p, q;
      for (a = 0; a < ids.length; a++) for (b = a + 1; b < ids.length; b++) {
        var acc = 0, cnt = 0;
        for (p = 0; p < n; p++) if (lab[p] === ids[a])
          for (q = 0; q < n; q++) if (lab[q] === ids[b]) { acc += cos(p, q); cnt++; }
        var dd = cnt ? acc / cnt : Infinity;
        if (dd < best) { best = dd; ia = ids[a]; ib = ids[b]; }
      }
      if (ia < 0 || (new Set(lab).size <= maxK && best > TH)) break;
      if (best > 0.55) break;
      for (p = 0; p < n; p++) if (lab[p] === ib) lab[p] = ia;
    }
    var map = {}, out = [], k = 0;
    for (var i = 0; i < n; i++) {
      if (map[lab[i]] == null) map[lab[i]] = String(++k);
      out.push(map[lab[i]]);
    }
    return out;
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

  async function diarizarProfesional(blob, cues, maxK) {
    var pcm = await pcm16k(blob);
    if (!pcm) throw new Error('audio');
    var dur = pcm.length / 16000;
    var segs = [];
    try {
      var out = await correrSeg(pcm);
      segs = logitsATurnos(out, dur);
    } catch (_) {
      segs = vadTurnos(pcm, 16000);
    }
    if (!segs.length && cues && cues.length) {
      segs = cues.map(function (c) { return { t0: c.t0, t1: c.t1, hablante: '1' }; });
    }
    var vecs = [], keep = [];
    for (var i = 0; i < segs.length; i++) {
      try {
        var v = await embedTurno(pcm, segs[i].t0, segs[i].t1);
        if (v) { vecs.push(v); keep.push(segs[i]); }
      } catch (_) {}
    }
    if (!vecs.length) throw new Error('sin embeddings');
    var labs = clusterEmb(vecs, maxK);
    return keep.map(function (s, k) {
      return { t0: s.t0, t1: s.t1, hablante: labs[k] || '1' };
    });
  }

  async function diarizar(blob, opciones) {
    opciones = opciones || {};
    var maxK = Math.max(2, Math.min(5, +opciones.maxK || 3));
    var cues = opciones.cues || [];
    var modo = opciones.modo || 'auto';
    var variante = opciones.variante || activa || 'fp16';
    if (modo === 'ligero' || modo === 'rapido' || modo === 'off') {
      return { turnos: await diarizarLigero(blob, cues, maxK), modo: 'ligero' };
    }
    try {
      await cargar({ onProgreso: onProgreso, variante: variante });
      var turnos = await diarizarProfesional(blob, cues, maxK);
      if (!turnos.length && cues.length) {
        return { turnos: await diarizarLigero(blob, cues, maxK), modo: 'ligero', aviso: 'sin-turnos' };
      }
      return { turnos: turnos, modo: 'profesional', variante: variante, origen: urlModelo('') };
    } catch (e) {
      ultimoError = String(e.message || e);
      if (modo === 'profesional' && opciones.fallback === false) throw e;
      return { turnos: await diarizarLigero(blob, cues, maxK), modo: 'ligero', aviso: ultimoError };
    }
  }

  function solape(a0, a1, b0, b1) {
    return Math.max(0, Math.min(a1, b1) - Math.max(a0, b0));
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

  async function borrar(v) {
    if (v && VARIANTES[v]) {
      var p = pack(v);
      await borrarArchivo(p.seg);
      sesiones[v] = { seg: null, emb: null };
      var otra = v === 'fp16' ? 'fp32' : 'fp16';
      if (!(await hayCache(otra))) {
        await borrarArchivo(p.emb);
        sesiones.fp16.emb = null;
        sesiones.fp32.emb = null;
        sesEmb = null;
      }
      if (activa === v) sesSeg = null;
    } else {
      sesSeg = null; sesEmb = null;
      sesiones = { fp16: { seg: null, emb: null }, fp32: { seg: null, emb: null } };
      try {
        if (navigator.storage && navigator.storage.getDirectory) {
          var root = await navigator.storage.getDirectory();
          try { await root.removeEntry('diar-engine', { recursive: true }); } catch (_) {}
        }
      } catch (_) {}
    }
    setEstado('no-descargado');
    return { ok: true };
  }

  (async function probe() {
    try { if (await hayCache()) setEstado('cache'); } catch (_) {}
  })();

  global.DiarizationEngine = {
    cargar: cargar,
    diarizar: diarizar,
    fusionar: fusionar,
    borrar: borrar,
    estado: function () { return estado; },
    error: function () { return ultimoError; },
    modelo: function () { return urlModelo(''); },
    mb: function (v) { return pack(v || activa).mb; },
    hayCache: hayCache,
    serverBase: serverBase,
    variantes: function () { return VARIANTES; }
  };
})(typeof window !== 'undefined' ? window : self);
