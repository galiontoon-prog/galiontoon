SUBIR A GITHUB (20260914f)
==========================
Obligatorios:
  1. Grabadora-Subtitulos.html   (motores Whisper + Moonshine + Diarización YA VAN DENTRO)
  2. index.html
  3. whisper-sw.js               (el service worker NO puede ir dentro del HTML)

Ya NO hace falta subir:
  - WhisperEngine.js
  - MoonshineEngine.js
  - DiarizationEngine.js

Los modelos .onnx siguen en GitHub Releases / Hugging Face. No van en el HTML.

Prueba:
  .../Grabadora-Subtitulos.html?v=20260914f
  Ctrl+Shift+R
  Consola: [Grabadora] versión 20260914f motores embebidos Whisper true Moonshine true Diar true
