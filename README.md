# Yui NLP

> [!WARNING]
> Este es un modulo interno del proyecto Yui. No esta diseñado para uso general. Si quieres usarlo, vas a necesitar adaptar las keywords y patrones a tu propio asistente de voz.

**yui_nlp** es un gatekeeper NLP escrito en Rust que clasifica texto transcrito por Whisper STT como **comando** o **conversacion casual**. Actua como un filtro ultrarapido antes de evaluar los regex de Python, eliminando falsos positivos y ahorrando ciclos de CPU.

## Como Funciona

```
Whisper STT → yui_nlp.classify(texto) → ¿Es comando? → Regex Python
                                       → ¿Es chat?   → LLM directo
```

El clasificador usa 3 capas de decision:

1. **Señales negativas** — Detecta negaciones ("no busques"), usos coloquiales ("me voy a poner a jugar") y preguntas personales ("¿quién es tu creador?")
2. **Keywords de comando** — Busca verbos imperativos de las herramientas actuales usando automatas Aho-Corasick en una sola pasada
3. **Contexto y posicion** — Valida que la keyword este en posicion de comando (cerca del inicio) y no sea narrativa

## API Python

```python
import yui_nlp

# Clasificacion completa
result = yui_nlp.classify("busca el precio del dolar")
print(result.is_command)   # True
print(result.category)     # "web_search"
print(result.confidence)   # 0.85

# Version rapida (solo bool)
if yui_nlp.is_command("abre spotify"):
    # Evaluar regex
    pass
```

### Categorias

| Categoria | Descripcion | Ejemplo |
|-----------|-------------|---------|
| `open_app` | Abrir aplicaciones | "abre spotify", "ejecuta translator++" |
| `web_search` | Buscar en internet | "busca el dolar", "qué es markdown" |
| `reminder` | Crear recordatorios | "recuérdame en 5 minutos", "pon una alarma" |
| `mode` | Cambiar modo del LLM | "modo rendimiento", "modo local" |

## Requisitos

- Rust 1.70+ (solo para compilar)
- Python 3.8+
- Maturin (`pip install maturin`)

## Compilar

```powershell
# Desarrollo (instala directo en el venv activo)
maturin develop --release

# Generar .whl para distribuir
maturin build --release
# El .whl se genera en target/wheels/
```

## Estructura

```
yui-nlp/
├── Cargo.toml           # Dependencias Rust
├── pyproject.toml        # Configuracion Maturin
├── src/
│   ├── lib.rs            # Binding PyO3 (expone classify/is_command a Python)
│   ├── classifier.rs     # Motor de clasificacion (3 capas)
│   └── normalizer.rs     # Normalizacion texto español (acentos, puntuacion)
└── .gitignore
```

## Stack

| Componente | Tecnologia |
|------------|------------|
| Motor de clasificacion | Rust + Aho-Corasick |
| Binding Python | PyO3 0.23 |
| Build system | Maturin |
| Normalizacion | unicode-normalization |

## Tests

```powershell
cargo test
```

Los tests cubren:
- Comandos validos de cada categoria (abrir apps, buscar, recordatorios, modo)
- Falsos positivos documentados (coloquialismos, negaciones, preguntas personales)
- Conversacion casual sin keywords

## Licencia

MIT License - Creado por **EDAKZIN**

---
**Version 0.2.0**
