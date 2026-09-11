DESPLIEGUE — Grabadora + Subtítulos (admin)
===========================================
Usuario final = URL HTTPS. Cero config.

Archivos a subir (misma carpeta, mismos nombres)
------------------------------------------------
  Grabadora-Subtitulos.html
  WhisperEngine.js
  (opcional) README-DESPLIEGUE.txt  ← no lo enlaces en Pages

Carpeta de repo (ejemplo)
-------------------------
  galiontoon/g-editor/
    Grabadora-Subtitulos.html
    WhisperEngine.js

GitHub Pages
------------
  Settings → Pages → Source: Deploy from a branch
  Branch: main  /  Folder: / (root)  o  /docs
  Si los archivos viven en galiontoon/g-editor/:
    URL típica:
    https://<user>.github.io/galiontoon/g-editor/Grabadora-Subtitulos.html
  Esperar el tilde verde de Pages. Forzar HTTPS (Settings → Pages).

Bluehost / Apache
-----------------
  MIME extra si el host no los trae:

    AddType application/wasm .wasm
    AddType application/octet-stream .onnx
    AddType application/javascript .js

  HTTPS: panel → SSL / Let's Encrypt. Sin https el mic
  y getDisplayMedia fallan en Chrome.

Verificar
---------
  1. Abrir la URL https://…/Grabadora-Subtitulos.html
  2. Consola: cero 404 de ./WhisperEngine.js
  3. «Descargar modelo» → barra de progreso → «Modelo listo»
  4. Recargar: no vuelve a bajar 75 MB
  5. Candado del navegador = seguro

Borrar modelo (limpieza)
------------------------
  En la propia página: botón «Borrar modelo».
  O DevTools → Application → Storage → Clear site data
  (Cache Storage + OPFS / File System del origen).

Rutas
-----
  HTML carga el motor con:  ./WhisperEngine.js
  El modelo NO se sube al repo. Sale de Hugging Face
  la primera vez y queda en el navegador del usuario.

No subir
--------
  G-Editor.html (fuera de este paquete)
  iniciar.bat / iniciar.command
  TextTool.html
