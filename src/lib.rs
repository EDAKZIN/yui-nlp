/// Modulo principal de yui_nlp.
/// Expone la funcion classify() a Python via PyO3.

mod normalizer;
mod classifier;

use pyo3::prelude::*;
use classifier::Classifier;
use std::sync::OnceLock;

/// Instancia global del clasificador (se inicializa una sola vez).
/// Los automatas Aho-Corasick se compilan en el primer uso.
static CLASSIFIER: OnceLock<Classifier> = OnceLock::new();

/// Obtiene o inicializa el clasificador global.
fn get_classifier() -> &'static Classifier {
    CLASSIFIER.get_or_init(Classifier::new)
}

// ---------------------------------------------------------------------------
// Clases expuestas a Python
// ---------------------------------------------------------------------------

/// Resultado de clasificacion expuesto a Python.
/// Contiene si es comando, la categoria detectada y la confianza.
#[pyclass]
#[derive(Clone)]
struct ClassifyResult {
    #[pyo3(get)]
    is_command: bool,
    #[pyo3(get)]
    category: Option<String>,
    #[pyo3(get)]
    confidence: f32,
}

#[pymethods]
impl ClassifyResult {
    fn __repr__(&self) -> String {
        match &self.category {
            Some(cat) => format!(
                "ClassifyResult(is_command={}, category='{}', confidence={:.2})",
                self.is_command, cat, self.confidence
            ),
            None => format!(
                "ClassifyResult(is_command={}, category=None, confidence={:.2})",
                self.is_command, self.confidence
            ),
        }
    }

    fn __bool__(&self) -> bool {
        self.is_command
    }
}

// ---------------------------------------------------------------------------
// Funciones expuestas a Python
// ---------------------------------------------------------------------------

/// Clasifica un texto como comando o conversacion casual.
///
/// Parametros:
///     text: Texto transcrito por Whisper STT.
///
/// Retorna:
///     ClassifyResult con is_command, category y confidence.
///
/// Ejemplo en Python:
///     import yui_nlp
///     result = yui_nlp.classify("busca el precio del dolar")
///     if result.is_command:
///         print(f"Comando detectado: {result.category}")
#[pyfunction]
fn classify(text: &str) -> ClassifyResult {
    let classifier = get_classifier();
    let result = classifier.classify(text);

    ClassifyResult {
        is_command: result.is_command,
        category: result.category.map(|c| c.as_str().to_string()),
        confidence: result.confidence,
    }
}

/// Verifica rapido si un texto parece comando (solo retorna bool).
/// Version simplificada de classify() para uso directo.
#[pyfunction]
fn is_command(text: &str) -> bool {
    let classifier = get_classifier();
    classifier.classify(text).is_command
}

// ---------------------------------------------------------------------------
// Definicion del modulo Python
// ---------------------------------------------------------------------------

/// Modulo yui_nlp: gatekeeper NLP para clasificacion de comandos.
#[pymodule]
fn yui_nlp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(classify, m)?)?;
    m.add_function(wrap_pyfunction!(is_command, m)?)?;
    m.add_class::<ClassifyResult>()?;
    Ok(())
}
