# viral-clipper

Convierte un video largo (YouTube, VOD de stream, podcast o archivo local) en clips verticales 9:16 listos para TikTok, Reels y Shorts. Usa Whisper para transcribir, Claude para escoger los momentos y Remotion para la edición.

Está pensado para la retención actual: mucha información visual, ritmo constante y un estímulo nuevo cada segundo.

## Qué hace

| Paso | Qué pasa |
|---|---|
| `fetch` | Descarga con **yt-dlp** (≤1080p). Si existen, también trae el heatmap "lo más visto" de YouTube y el chat de la transmisión. |
| `audio` | Saca el audio a 16 kHz y mide el volumen cada 0.5 s. |
| `transcribe` | **faster-whisper** transcribe con tiempos por palabra, en bloques de 10 min. Si se corta, sigue donde iba, así que aguanta un stream de 5 h. Usa la GPU NVIDIA si la encuentra. |
| `score` | Pasa ventanas deslizantes y las califica por picos de volumen (gritos, risas), ritmo de habla, palabras de hype, heatmap y ráfagas de chat. |
| `pick` | **Claude** (`claude -p`) elige los mejores momentos. Cada uno arranca en la frase más fuerte, se entiende solo y termina en remate. También escribe el título gancho, las keywords, los emojis y la descripción para publicar. |
| `cut` | Corta con ffmpeg y sigue la cara con **YuNet** (OpenCV DNN), detectando los cambios de escena. Escoge el formato solo y detecta y oculta los subtítulos quemados del video original. |
| `render` | **Remotion** arma el clip final. |

### Estilo de edición

- **Jump cuts** en cada pausa de más de 0.35 s.
- **Cambio de zoom** en casi cada corte, más un empuje lento continuo.
- **Golpe de zoom con sacudida y destello** en los momentos más fuertes.
- **Subtítulos de 1 a 3 palabras** que entran con rebote: la palabra activa en amarillo y las keywords en verde, más grandes.
- **Emojis animados** con sonido "pop" cuando se dice la palabra.
- **Título gancho** desde el primer cuadro y **barra de progreso** arriba.

### Formatos verticales (`--layout auto` escoge solo)

- `face`: recorte vertical que sigue al que habla.
- `stream`: facecam arriba y juego o pantalla abajo.
- `fit`: toma abierta grande y centrada sobre un fondo desenfocado de sí misma, para escenas con varias personas.

## Requisitos

- Python 3.10+ y Node 18+.
- [Claude Code](https://claude.com/claude-code) instalado y con sesión iniciada; el comando `claude` debe estar en el PATH. Es lo que usa el paso `pick`.
- No necesitas ffmpeg aparte: `imageio-ffmpeg` trae su propio binario.
- Opcional: GPU NVIDIA con `nvidia-cublas-cu12` y `nvidia-cudnn-cu12` para transcribir mucho más rápido.

```bash
pip install -r requirements.txt
npm install
```

## Uso

```bash
python tools/clipper.py "https://www.youtube.com/watch?v=XXXX" --id mi-video --n 6
```

Los clips salen en `out/clips/<id>/cNN-titulo.mp4`. Cada uno trae un `.txt` con el título, la descripción con hashtags y por qué se eligió.

| Opción | Default | |
|---|---|---|
| `--n` | 6 | cuántos clips |
| `--min` / `--max` | 20 / 55 | duración en segundos |
| `--layout` | auto | `face`, `stream`, `fit` |
| `--model` | small | modelo de Whisper (`base` va más rápido, `medium` es más preciso) |
| `--lang` | es | idioma del audio |
| `--from-step` | — | repetir desde un paso, p. ej. `pick` para volver a escoger sin re-transcribir |
| `--no-render` | — | parar antes de renderizar |
| `--handle` | — | @usuario que aparece abajo del clip |
| `--accent` | #FFE600 | color de acento |

Todo queda en caché en `content/clips/<id>/`, y cada paso se salta si ya existe su salida.

Para ajustar el diseño en vivo: `npm run studio` y abre la composición `ViralClip` con los props de `content/clips/<id>/c01.props.json`.

## Tiempos de referencia

En una PC con Ryzen 3 4350G y GTX 1060 3 GB:

- Transcribir 22 min de video en GPU toma unos 3 min.
- Renderizar un clip de unos 30 s toma unos 3.5 min.

## Notas

- Recorta solo contenido tuyo o con permiso del creador. Muchos streamers tienen programas de clippers. Subir clips de otros sin permiso puede traer reclamos de copyright.
- El modelo YuNet se descarga solo la primera vez desde [opencv_zoo](https://github.com/opencv/opencv_zoo) (licencia MIT).
- Remotion tiene su propia [licencia](https://www.remotion.dev/license): es gratis para personas y empresas pequeñas.

## Licencia

MIT
