/// Modulo de normalizacion de texto en español.
/// Maneja acentos, variaciones de Whisper y limpieza general.

use unicode_normalization::UnicodeNormalization;

/// Mapa de caracteres acentuados a su version sin acento.
/// Se usa para matching flexible sin perder el texto original.
const ACCENT_MAP: &[(char, char)] = &[
    ('á', 'a'), ('é', 'e'), ('í', 'i'), ('ó', 'o'), ('ú', 'u'),
    ('ü', 'u'), ('ñ', 'n'),
];

/// Normaliza texto para matching: minusculas, sin acentos, sin puntuacion extra.
/// No modifica el texto original del usuario, solo se usa internamente.
pub fn normalize(text: &str) -> String {
    let lowered = text.to_lowercase();

    // Normalizar unicode (NFC) y quitar acentos
    let normalized: String = lowered
        .nfc()
        .map(|c| {
            ACCENT_MAP
                .iter()
                .find(|(from, _)| *from == c)
                .map(|(_, to)| *to)
                .unwrap_or(c)
        })
        .collect();

    // Quitar signos de puntuacion que no aportan al matching
    let cleaned: String = normalized
        .chars()
        .map(|c| match c {
            '¿' | '?' | '¡' | '!' | ',' | '.' | ';' | ':' => ' ',
            _ => c,
        })
        .collect();

    // Colapsar espacios multiples y hacer trim
    collapse_spaces(&cleaned)
}

/// Colapsa espacios consecutivos en uno solo.
fn collapse_spaces(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_space = true; // Empieza en true para hacer trim izquierdo

    for c in text.chars() {
        if c == ' ' {
            if !prev_space {
                result.push(' ');
            }
            prev_space = true;
        } else {
            result.push(c);
            prev_space = false;
        }
    }

    // Trim derecho
    if result.ends_with(' ') {
        result.pop();
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_acentos() {
        assert_eq!(normalize("búscame"), "buscame");
        assert_eq!(normalize("recuérdame"), "recuerdame");
        assert_eq!(normalize("quién es"), "quien es");
    }

    #[test]
    fn test_normalize_puntuacion() {
        assert_eq!(normalize("¿qué es eso?"), "que es eso");
        assert_eq!(normalize("¡busca eso!"), "busca eso");
    }

    #[test]
    fn test_normalize_espacios() {
        assert_eq!(normalize("  busca   el   precio  "), "busca el precio");
    }

    #[test]
    fn test_normalize_mayusculas() {
        assert_eq!(normalize("Yui BUSCA eso"), "yui busca eso");
    }
}
