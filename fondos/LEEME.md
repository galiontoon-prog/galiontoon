# Fondos de vídeo de Radio Galion

Estos vídeos **ya no viajan dentro de `galiontoon.html`**. Desde la v1.58 se piden al
elegir el tema.

## Por qué

Medido sobre el archivo completo:

```
los tres fondos      6 578 485 bytes crudos  ·  4 486 464 comprimidos
                     el 27 % de todo lo que se descargaba
y solo se ve UNO cada vez
```

El archivo bajó de **16,5 MB comprimidos a 12,1**.

## Cómo publicarlos

1. Crea un repositorio público en GitHub, por ejemplo `TUUSUARIO/galiontoon`.
2. Sube esta carpeta `fondos/` tal cual.
3. En el archivo, cambia esta línea por tu usuario:

```js
CDN_FONDOS: 'https://cdn.jsdelivr.net/gh/USUARIO/galiontoon@main/fondos/',
```

**No uses `raw.githubusercontent.com`.** GitHub lo desaconseja expresamente como CDN y
limita el ancho de banda. jsDelivr sirve desde tu repositorio, es gratis y sin límite —y es
lo que ya usa el programa para el modelo de recorte.

## Para añadir un fondo nuevo

1. Deja el `.mp4` en esta carpeta y súbelo al repositorio.
2. Añade una línea a `themes` en el archivo:

```js
{ name: 'Mi fondo nuevo', video: 'mi-fondo.mp4',
  bg: 'linear-gradient(120deg, #10061f, #2a1150, #06010f, #3d0a2e)' },
```

El `bg` **no es decorativo**: es lo que se ve si el vídeo no llega. Ponle colores que peguen
con el vídeo.

## Qué pasa si el CDN falla

Nada grave, y está comprobado sin conexión: el vídeo se esconde y queda el degradado
animado, que es lo que ya usan tres de los seis temas. **La radio suena igual.** El fondo es
decoración.

## Consejo de tamaño

Comprime antes de subir. Los que hay:

```
degradado.mp4        69 852 bytes
cueva-sirena.mp4    839 043 bytes
fiesta-dj.mp4     3 539 877 bytes   <- este es el que más pesa
```

Un fondo de pantalla completa a 1280×720 y 20 fotogramas por segundo con `-crf 30` suele
bajar de 1 MB sin que se note.
