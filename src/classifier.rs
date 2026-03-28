/// Motor de clasificacion de texto.
/// Decide si un texto transcrito por Whisper es un comando o conversacion casual.
///
/// Arquitectura de 3 capas:
///   1. Señales negativas (bloqueo inmediato)
///   2. Keywords de comando por categoria (deteccion)
///   3. Decision final basada en posicion y contexto

use aho_corasick::AhoCorasick;
use crate::normalizer;

// ---------------------------------------------------------------------------
// Categorias de comando que el clasificador puede detectar
// ---------------------------------------------------------------------------

/// Categoria de comando detectada por el clasificador.
#[derive(Debug, Clone, PartialEq)]
pub enum Category {
    OpenApp,
    WebSearch,
    Reminder,
    Mode,
}

impl Category {
    /// Representacion de texto para exponer a Python.
    pub fn as_str(&self) -> &'static str {
        match self {
            Category::OpenApp => "open_app",
            Category::WebSearch => "web_search",
            Category::Reminder => "reminder",
            Category::Mode => "mode",
        }
    }
}

/// Resultado de la clasificacion.
#[derive(Debug, Clone)]
pub struct ClassifyResult {
    pub is_command: bool,
    pub category: Option<Category>,
    pub confidence: f32,
}

// ---------------------------------------------------------------------------
// Keywords organizadas por categoria
// Heredadas de los regex actuales en Python (yui_assistant.py, reminders.py)
// ---------------------------------------------------------------------------

/// Keywords de apertura de aplicaciones (yui_assistant.py L337-340).
/// Incluye: imperativo, subjuntivo, infinitivo y formas pronominales.
const OPEN_KEYWORDS: &[&str] = &[
    // Imperativo directo
    "abre ", "abreme ", "abrelo ", "abrela ",
    // Infinitivo y subjuntivo (estructuras indirectas: "necesito que abras X")
    "abrir ", "abras ", "abrirme ",
    // Pasado inmediato como orden informal ("abri chrome" = "abrí chrome")
    "abri ",
    // Ejecutar
    "ejecuta ", "ejecutar ", "ejecutes ",
    // Iniciar
    "inicia ", "iniciar ", "inicies ",
];

/// Keywords ambiguas de apertura.
/// "pon/poner/ponme" son verbos extremadamente comunes en español coloquial.
/// Se tratan aparte con validacion extra de contexto.
const OPEN_KEYWORDS_AMBIGUOUS: &[&str] = &[
    "pon ", "poner ", "ponme ",
];

/// Keywords de busqueda web (yui_assistant.py L370-377).
/// Incluye imperativos, subjuntivos e infinitivos.
const SEARCH_KEYWORDS: &[&str] = &[
    // Buscar: imperativo, subjuntivo, infinitivo
    "busca ", "buscas ", "buscar ", "busques ", "buscame ", "buscalo ",
    // Investigar
    "investiga ", "investigar ", "investigues ",
    // Frases completas de solicitud de informacion
    "dime sobre ", "hablame de ", "hablame sobre ",
    "cuentame sobre ", "cuentame de ",
    "informacion sobre ", "informacion de ",
];

/// Keywords de busqueda tipo pregunta (requieren validacion extra).
/// Patrones "que es X" y "quien es X" pueden ser retoricos o personales.
const SEARCH_QUESTION_KEYWORDS: &[&str] = &[
    "que es ", "que son ", "que significa ",
    "quien es ", "quien fue ", "quien era ",
    "que paso con ", "que ocurrio con ",
];

/// Keywords de recordatorios (reminders.py L132-139).
/// Frases muy especificas: rara vez generan falsos positivos.
const REMINDER_KEYWORDS: &[&str] = &[
    "recuerdame", "recordarme",
    "avisame",
    "hazme un recordatorio", "pon un recordatorio", "ponme un recordatorio",
    "pon una alarma", "ponme una alarma", "pon un alarma", "ponme un alarma",
    "temporizador", "pon un timer",
];

/// Keywords de modo rendimiento/local (yui_assistant.py L327-334).
/// Frases multi-palabra muy especificas, practicamente sin ambiguedad.
const MODE_KEYWORDS: &[&str] = &[
    "modo rendimiento", "modo de rendimiento",
    "modo rapido", "modo turbo",
    "activa groq", "usa groq",
    "modo local", "modo normal", "modo lento",
];

// ---------------------------------------------------------------------------
// Patrones de bloqueo (señales negativas)
// ---------------------------------------------------------------------------

/// Negaciones explicitas del usuario dirigidas a los verbos de comando.
/// Cubren variaciones naturales del español hablado.
const NEGATION_PATTERNS: &[&str] = &[
    // Negaciones directas con verbo de comando
    "no busques", "no buscar", "no la busques", "no lo busques",
    "no abras", "no abrir", "no lo abras", "no la abras",
    "no ejecutes", "no ejecutar",
    "no inicies", "no iniciar",
    "no investigues", "no investigar",
    "no pongas",
    // Negaciones con estructura "no quiero/necesito que..."
    "no quiero que busques", "no quiero que abras", "no quiero que ejecutes",
    "no necesito que busques", "no necesito que abras",
    // Negaciones con "no es necesario" / "no hace falta"
    "no es necesario buscar", "no es necesario abrir",
    "no hace falta buscar", "no hace falta abrir",
    // Negaciones indirectas / correcciones
    "no me busques", "no me abras",
    "deja de buscar", "deja de abrir",
    "para de buscar",
    "no era eso", "eso no",
];

/// Frases coloquiales donde los verbos de comando se usan figurativamente.
/// Son patrones de habla natural en español que no implican una accion.
const COLLOQUIAL_BLOCKERS: &[&str] = &[
    // === "poner" coloquial ===
    // Estructura "poner(se) a + infinitivo" = empezar a hacer algo
    "poner a ", "ponerme a ", "ponerse a ", "ponerte a ", "ponerle a ",
    "me pongo a ", "se pone a ", "te pones a ", "nos ponemos a ",
    "me voy a poner a ", "voy a ponerme a ", "vamos a ponernos a ",
    "me puse a ", "se puso a ", "se pusieron a ",
    // "ponerse" + adjetivo = cambiar de estado
    "me pongo ", "se pone ", "te pones ",
    "me puse ", "se puso ", "me he puesto ",
    // === "abrir" figurativo ===
    // Formas verbales que NO son imperativas
    "no te abro", "cuando no te abro", "de cuando no te abro",
    "te abro", "la abro", "lo abro", // "yo te abro" = narrativo
    "se abre", "se abrio", "se me abre", "se me abrio",
    "me abri", "ya abri", "cuando abri", // Pasado
    "quiero abrir", // Deseo, no orden directa
    // === "ejecutar" descriptivo ===
    "se ejecuta", "se ejecuto", "cuando se ejecuta",
    "esta ejecutando", "sigue ejecutando",
    // === "iniciar" descriptivo ===
    "se inicia", "se inicio", "cuando inicia", "cuando se inicia",
    "ya inicio", "esta iniciando", "sigue iniciando",
    // === "buscar" figurativo ===
    "lo que busca", "lo que busco", // Narrativo
    "para que seguir",
    "anda buscando", "sigue buscando", // Descriptivo
];

/// Frases "retorica / eso de" que indican pregunta casual, no busqueda.
/// Filtran patrones como "que es eso de X" que no piden info web.
const RHETORICAL_PATTERNS: &[&str] = &[
    "que es eso de ", "que es eso del ", "que es eso que ",
    "para que es eso", "como es eso de ", "como es eso que ",
    "y eso que es",
];

/// Contexto personal: preguntas dirigidas a Yui sobre si misma.
/// Bloquean busqueda web porque el LLM las responde del prompt.
const PERSONAL_CONTEXT: &[&str] = &[
    // Preguntas sobre identidad de Yui
    "tu creador", "tu creado", "creaste", "te creo",
    "quien te creo", "quien te hizo", "quien te programo",
    "eres tu", "eres yui", "tu nombre", "como te llamas",
    "fuiste creado", "fuiste creada", "fuiste hecho", "fuiste hecha",
    // Preguntas sobre el usuario desde la perspectiva de Yui
    "mi nombre", "como me llamo", "tu sabes",
    // Referencias a Edakzin (creador)
    "edakzin",
];

// ---------------------------------------------------------------------------
// Clasificador principal
// ---------------------------------------------------------------------------

/// Clasificador de texto que decide si una frase es comando o conversacion.
/// Pre-compila automatas Aho-Corasick para matching en una sola pasada.
pub struct Classifier {
    /// Automata para keywords directas de apertura
    open_ac: AhoCorasick,
    /// Automata para keywords ambiguas de apertura
    open_ambiguous_ac: AhoCorasick,
    /// Automata para keywords de busqueda
    search_ac: AhoCorasick,
    /// Automata para keywords de pregunta-busqueda
    search_question_ac: AhoCorasick,
    /// Automata para keywords de recordatorios
    reminder_ac: AhoCorasick,
    /// Automata para keywords de modo
    mode_ac: AhoCorasick,
    /// Automata para patrones de negacion
    negation_ac: AhoCorasick,
    /// Automata para frases coloquiales que bloquean
    colloquial_ac: AhoCorasick,
    /// Automata para patrones retoricos
    rhetorical_ac: AhoCorasick,
    /// Automata para contexto personal
    personal_ac: AhoCorasick,
}

impl Classifier {
    /// Crea un nuevo clasificador con todos los automatas pre-compilados.
    pub fn new() -> Self {
        Self {
            open_ac: AhoCorasick::new(OPEN_KEYWORDS).expect("Error compilando open keywords"),
            open_ambiguous_ac: AhoCorasick::new(OPEN_KEYWORDS_AMBIGUOUS).expect("Error compilando open ambiguous"),
            search_ac: AhoCorasick::new(SEARCH_KEYWORDS).expect("Error compilando search keywords"),
            search_question_ac: AhoCorasick::new(SEARCH_QUESTION_KEYWORDS).expect("Error compilando search question"),
            reminder_ac: AhoCorasick::new(REMINDER_KEYWORDS).expect("Error compilando reminder keywords"),
            mode_ac: AhoCorasick::new(MODE_KEYWORDS).expect("Error compilando mode keywords"),
            negation_ac: AhoCorasick::new(NEGATION_PATTERNS).expect("Error compilando negation patterns"),
            colloquial_ac: AhoCorasick::new(COLLOQUIAL_BLOCKERS).expect("Error compilando colloquial blockers"),
            rhetorical_ac: AhoCorasick::new(RHETORICAL_PATTERNS).expect("Error compilando rhetorical patterns"),
            personal_ac: AhoCorasick::new(PERSONAL_CONTEXT).expect("Error compilando personal context"),
        }
    }

    /// Clasifica un texto como comando o conversacion casual.
    ///
    /// Retorna un ClassifyResult con:
    /// - is_command: si vale la pena evaluar los regex de Python
    /// - category: la categoria detectada (open_app, web_search, reminder, mode)
    /// - confidence: nivel de confianza (0.0 a 1.0)
    pub fn classify(&self, text: &str) -> ClassifyResult {
        let normalized = normalizer::normalize(text);
        // Agregar espacio al final para que keywords con trailing space matcheen al final
        let padded = format!("{} ", normalized);

        let chat = ClassifyResult {
            is_command: false,
            category: None,
            confidence: 0.0,
        };

        // Capa 1: Señales de bloqueo
        let has_negation = self.negation_ac.is_match(&padded);
        let has_colloquial = self.colloquial_ac.is_match(&padded);
        let has_rhetorical = self.rhetorical_ac.is_match(&padded);

        // Capa 2: Keywords de comando por categoria (orden de prioridad)

        // --- Modo rendimiento/local (maxima prioridad, sin ambiguedad) ---
        if self.mode_ac.is_match(&padded) && !has_negation {
            return ClassifyResult {
                is_command: true,
                category: Some(Category::Mode),
                confidence: 0.95,
            };
        }

        // --- Recordatorios (alta especificidad) ---
        if self.reminder_ac.is_match(&padded) && !has_negation {
            return ClassifyResult {
                is_command: true,
                category: Some(Category::Reminder),
                confidence: 0.9,
            };
        }

        // --- Apertura de apps: keywords directas ---
        if self.open_ac.is_match(&padded) {
            if has_negation || has_colloquial {
                return chat;
            }
            // Verificar posicion: el verbo debe estar cerca del inicio de la frase
            // para descartar menciones narrativas al final
            if self.keyword_near_start(&padded, &self.open_ac, 40) {
                return ClassifyResult {
                    is_command: true,
                    category: Some(Category::OpenApp),
                    confidence: 0.85,
                };
            }
            // Keyword existe pero esta muy lejos del inicio: menor confianza
            return ClassifyResult {
                is_command: true,
                category: Some(Category::OpenApp),
                confidence: 0.6,
            };
        }

        // --- Apertura de apps: keywords ambiguas (pon/poner/ponme) ---
        if self.open_ambiguous_ac.is_match(&padded) {
            if has_negation || has_colloquial {
                return chat;
            }
            // "pon" ambiguo solo si esta al inicio
            if self.keyword_near_start(&padded, &self.open_ambiguous_ac, 20) {
                return ClassifyResult {
                    is_command: true,
                    category: Some(Category::OpenApp),
                    confidence: 0.55,
                };
            }
            return chat;
        }

        // --- Busqueda web: keywords directas ---
        if self.search_ac.is_match(&padded) {
            if has_negation {
                return chat;
            }
            return ClassifyResult {
                is_command: true,
                category: Some(Category::WebSearch),
                confidence: 0.85,
            };
        }

        // --- Busqueda web: preguntas tipo "que es X", "quien es X" ---
        if self.search_question_ac.is_match(&padded) {
            if has_negation || has_rhetorical {
                return chat;
            }
            // Bloquear preguntas personales dirigidas a Yui
            if self.personal_ac.is_match(&padded) {
                return chat;
            }
            return ClassifyResult {
                is_command: true,
                category: Some(Category::WebSearch),
                confidence: 0.7,
            };
        }

        // Capa 3: Sin keywords de comando detectadas
        chat
    }

    /// Verifica si alguna keyword del automata aparece dentro de los primeros N caracteres.
    /// Ayuda a distinguir "abre spotify" (comando) de "hoy cuando abri spotify" (narrativo).
    fn keyword_near_start(&self, text: &str, ac: &AhoCorasick, max_pos: usize) -> bool {
        for mat in ac.find_iter(text) {
            if mat.start() <= max_pos {
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn classifier() -> Classifier {
        Classifier::new()
    }

    // === Comandos validos (deben pasar como is_command=true) ===

    #[test]
    fn test_cmd_abrir_app() {
        let c = classifier();
        let r = c.classify("yui ejecuta translator++");
        assert!(r.is_command);
        assert_eq!(r.category.unwrap(), Category::OpenApp);
    }

    #[test]
    fn test_cmd_abrir_app_abre() {
        let c = classifier();
        let r = c.classify("abre spotify");
        assert!(r.is_command);
        assert_eq!(r.category.unwrap(), Category::OpenApp);
    }

    #[test]
    fn test_cmd_buscar_directo() {
        let c = classifier();
        let r = c.classify("busca el precio del dólar en perú");
        assert!(r.is_command);
        assert_eq!(r.category.unwrap(), Category::WebSearch);
    }

    #[test]
    fn test_cmd_buscar_que_es() {
        let c = classifier();
        let r = c.classify("yui busca qué es markdown");
        assert!(r.is_command);
        assert_eq!(r.category.unwrap(), Category::WebSearch);
    }

    #[test]
    fn test_cmd_buscar_internet() {
        let c = classifier();
        let r = c.classify("busca en internet qué se celebra hoy");
        assert!(r.is_command);
        assert_eq!(r.category.unwrap(), Category::WebSearch);
    }

    #[test]
    fn test_cmd_buscar_quien_es() {
        let c = classifier();
        let r = c.classify("yui busca quién es el actual presidente de perú");
        assert!(r.is_command);
        assert_eq!(r.category.unwrap(), Category::WebSearch);
    }

    #[test]
    fn test_cmd_que_es_twitter() {
        let c = classifier();
        let r = c.classify("yui, qué es twitter");
        assert!(r.is_command);
        assert_eq!(r.category.unwrap(), Category::WebSearch);
    }

    #[test]
    fn test_cmd_investiga() {
        let c = classifier();
        let r = c.classify("investiga qué se celebra hoy");
        assert!(r.is_command);
        assert_eq!(r.category.unwrap(), Category::WebSearch);
    }

    #[test]
    fn test_cmd_recordatorio() {
        let c = classifier();
        let r = c.classify("hazme un recordatorio en 2 minutos de pararme");
        assert!(r.is_command);
        assert_eq!(r.category.unwrap(), Category::Reminder);
    }

    #[test]
    fn test_cmd_alarma() {
        let c = classifier();
        let r = c.classify("yui pon una alarma para dos minutos de revisar opera");
        assert!(r.is_command);
        assert_eq!(r.category.unwrap(), Category::Reminder);
    }

    #[test]
    fn test_cmd_recuerdame() {
        let c = classifier();
        let r = c.classify("yui recuérdame en 5 minutos pararme");
        assert!(r.is_command);
        assert_eq!(r.category.unwrap(), Category::Reminder);
    }

    #[test]
    fn test_cmd_modo_rendimiento() {
        let c = classifier();
        let r = c.classify("activa modo rendimiento");
        assert!(r.is_command);
        assert_eq!(r.category.unwrap(), Category::Mode);
    }

    #[test]
    fn test_cmd_modo_local() {
        let c = classifier();
        let r = c.classify("modo local");
        assert!(r.is_command);
        assert_eq!(r.category.unwrap(), Category::Mode);
    }

    #[test]
    fn test_cmd_dime_sobre() {
        let c = classifier();
        let r = c.classify("dime sobre la segunda guerra mundial");
        assert!(r.is_command);
        assert_eq!(r.category.unwrap(), Category::WebSearch);
    }

    // === Falsos positivos (deben ser bloqueados: is_command=false) ===

    #[test]
    fn test_fp_poner_coloquial_jugar() {
        let c = classifier();
        let r = c.classify("creo que me voy a poner a jugar");
        assert!(!r.is_command, "Poner a + verbo es coloquial, no comando");
    }

    #[test]
    fn test_fp_ponerme_a_hacer() {
        let c = classifier();
        let r = c.classify("me voy a ponerme a hacer la tarea");
        assert!(!r.is_command, "Ponerme a + verbo es coloquial");
    }

    #[test]
    fn test_fp_me_pongo_a_estudiar() {
        let c = classifier();
        let r = c.classify("ya me pongo a estudiar no te preocupes");
        assert!(!r.is_command, "Me pongo a + verbo es coloquial");
    }

    #[test]
    fn test_fp_se_puso_a_llorar() {
        let c = classifier();
        let r = c.classify("se puso a llorar de la nada");
        assert!(!r.is_command, "Se puso a + verbo es narrativo");
    }

    #[test]
    fn test_fp_abro_figurativo() {
        let c = classifier();
        let r = c.classify("no sé de cuándo no te abro pero al menos para que sepas");
        assert!(!r.is_command, "'abro' figurativo no es comando");
    }

    #[test]
    fn test_fp_negacion_buscar() {
        let c = classifier();
        let r = c.classify("a mi no me gusta el fútbol, no busques eso");
        assert!(!r.is_command, "Negacion explicita 'no busques'");
    }

    #[test]
    fn test_fp_pregunta_creador() {
        let c = classifier();
        let r = c.classify("¿quién es tu creador?");
        assert!(!r.is_command, "Pregunta personal dirigida a Yui");
    }

    #[test]
    fn test_fp_quien_te_creo() {
        let c = classifier();
        let r = c.classify("¿por quién fuiste creado?");
        assert!(!r.is_command, "Pregunta personal sobre el creador de Yui");
    }

    #[test]
    fn test_fp_que_es_eso_de() {
        let c = classifier();
        let r = c.classify("qué es eso de 20 preguntas");
        assert!(!r.is_command, "'que es eso de X' es retorica");
    }

    #[test]
    fn test_fp_se_ejecuta_descriptivo() {
        let c = classifier();
        let r = c.classify("se ejecuta muy lento el programa");
        assert!(!r.is_command, "'se ejecuta' es descriptivo, no comando");
    }

    #[test]
    fn test_fp_se_inicia_descriptivo() {
        let c = classifier();
        let r = c.classify("cuando se inicia la computadora tarda mucho");
        assert!(!r.is_command, "'se inicia' es descriptivo");
    }

    #[test]
    fn test_fp_deja_de_buscar() {
        let c = classifier();
        let r = c.classify("deja de buscar eso yui");
        assert!(!r.is_command, "'deja de buscar' es negacion");
    }

    #[test]
    fn test_fp_lo_que_busco() {
        let c = classifier();
        let r = c.classify("eso no es lo que busco");
        assert!(!r.is_command, "'lo que busco' es narrativo");
    }

    #[test]
    fn test_fp_no_abras_correccion() {
        let c = classifier();
        let r = c.classify("no no no, no abras eso");
        assert!(!r.is_command, "Correccion con negacion");
    }

    #[test]
    fn test_fp_edakzin_personal() {
        let c = classifier();
        let r = c.classify("quién es edakzin");
        assert!(!r.is_command, "Pregunta sobre Edakzin es personal");
    }

    // === Conversacion normal (sin keywords, deben ser chat) ===

    #[test]
    fn test_chat_saludo() {
        let c = classifier();
        let r = c.classify("hola yui, cómo estás hoy");
        assert!(!r.is_command);
    }

    #[test]
    fn test_chat_pelicula() {
        let c = classifier();
        let r = c.classify("me gustó mucho la película de ayer");
        assert!(!r.is_command);
    }

    #[test]
    fn test_chat_opinion() {
        let c = classifier();
        let r = c.classify("no sé si debería ir al gimnasio mañana");
        assert!(!r.is_command);
    }

    #[test]
    fn test_chat_emocional() {
        let c = classifier();
        let r = c.classify("estoy un poco cansado hoy pero bien");
        assert!(!r.is_command);
    }

    // === Conjugaciones indirectas (deben pasar como comando) ===

    #[test]
    fn test_cmd_necesito_que_abras() {
        let c = classifier();
        let r = c.classify("yui, necesito que abras translator++");
        assert!(r.is_command, "'necesito que abras' es comando indirecto");
        assert_eq!(r.category.unwrap(), Category::OpenApp);
    }

    #[test]
    fn test_cmd_puedes_abrir() {
        let c = classifier();
        let r = c.classify("puedes abrir chrome por favor");
        assert!(r.is_command, "'puedes abrir' es solicitud de apertura");
        assert_eq!(r.category.unwrap(), Category::OpenApp);
    }

    #[test]
    fn test_cmd_ejecutes() {
        let c = classifier();
        let r = c.classify("necesito que ejecutes el programa");
        assert!(r.is_command, "'ejecutes' subjuntivo es comando");
        assert_eq!(r.category.unwrap(), Category::OpenApp);
    }

    #[test]
    fn test_cmd_abrelo() {
        let c = classifier();
        let r = c.classify("abrelo ya");
        assert!(r.is_command, "'abrelo' pronominal es comando");
        assert_eq!(r.category.unwrap(), Category::OpenApp);
    }

    #[test]
    fn test_cmd_buscalo() {
        let c = classifier();
        let r = c.classify("buscalo en internet");
        assert!(r.is_command, "'buscalo' pronominal es comando");
        assert_eq!(r.category.unwrap(), Category::WebSearch);
    }

    #[test]
    fn test_cmd_hablame_sobre() {
        let c = classifier();
        let r = c.classify("hablame sobre la inteligencia artificial");
        assert!(r.is_command, "'hablame sobre' es solicitud de info");
        assert_eq!(r.category.unwrap(), Category::WebSearch);
    }

    #[test]
    fn test_cmd_que_paso_con() {
        let c = classifier();
        let r = c.classify("que paso con el terremoto de ayer");
        assert!(r.is_command, "'que paso con' es pregunta factual");
        assert_eq!(r.category.unwrap(), Category::WebSearch);
    }
}
