"""
Internationalization for belief cards (US-11).

Translates belief card text to user's language while keeping sources in original.

Features:
- Auto-detect language from prompt
- Translate card text only (keep sources original)
- Fallback to English
- Simple translation support (can be extended with real translation service)
"""

from typing import Optional, Dict
from dataclasses import dataclass
import re


# ============================================================================
# Language Detection
# ============================================================================

# Simple language patterns (can be extended with langdetect library)
LANGUAGE_PATTERNS = {
    'pt': [
        r'\b(é|são|está|estão|foi|foram|será|você|ele|ela|muito|mais|menos)\b',
        r'\b(com|para|por|em|de|do|da|dos|das|no|na|nos|nas)\b',
    ],
    'es': [
        r'\b(es|son|está|están|fue|fueron|será|usted|él|ella|muy|más|menos)\b',
        r'\b(con|para|por|en|de|del|los|las|el|la)\b',
    ],
    'fr': [
        r'\b(est|sont|était|étaient|sera|vous|il|elle|très|plus|moins)\b',
        r'\b(avec|pour|par|dans|de|du|des|le|la|les)\b',
    ],
    'de': [
        r'\b(ist|sind|war|waren|wird|Sie|er|sie|sehr|mehr|weniger)\b',
        r'\b(mit|für|von|in|der|die|das|den|dem)\b',
    ],
}


def detect_language(text: str) -> str:
    """
    Detect language from text.

    Simple pattern-based detection. For production, use langdetect or similar.

    Args:
        text: Text to analyze

    Returns:
        Language code ('en', 'pt', 'es', etc.)
    """
    if not text or len(text.strip()) < 10:
        return 'en'

    text_lower = text.lower()

    # Score each language
    scores = {}
    for lang, patterns in LANGUAGE_PATTERNS.items():
        score = 0
        for pattern in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            score += len(matches)
        scores[lang] = score

    # Get best match
    if scores:
        best_lang = max(scores.items(), key=lambda x: x[1])
        if best_lang[1] > 2:  # At least 2 matches
            return best_lang[0]

    return 'en'  # Default to English


# ============================================================================
# Translation Templates
# ============================================================================

@dataclass
class TranslationTemplate:
    """
    Translation template for belief cards.

    Attributes:
        lang_code: Language code
        belief_prefix: Prefix for belief cards
        confidence_label: Label for confidence
        updated_label: Label for update date
        sources_label: Label for sources
        context_label: Label for context
        tension_prefix: Prefix for tension annotations
    """
    lang_code: str
    belief_prefix: str = "[BELIEF]"
    confidence_label: str = "Confidence"
    updated_label: str = "Updated"
    sources_label: str = "Sources"
    context_label: str = "Context"
    tension_prefix: str = "⚠️"
    header: str = "# Relevant Beliefs"
    tension_header: str = "# ⚠️ Tensions (Contradictions)"


# Default templates
TRANSLATION_TEMPLATES = {
    'en': TranslationTemplate(
        lang_code='en',
        belief_prefix="[BELIEF]",
        confidence_label="Confidence",
        updated_label="Updated",
        sources_label="Sources",
        context_label="Context",
        header="# Relevant Beliefs",
        tension_header="# ⚠️ Tensions (Contradictions)"
    ),

    'pt': TranslationTemplate(
        lang_code='pt',
        belief_prefix="[CRENÇA]",
        confidence_label="Confiança",
        updated_label="Atualizado",
        sources_label="Fontes",
        context_label="Contexto",
        header="# Crenças Relevantes",
        tension_header="# ⚠️ Tensões (Contradições)"
    ),

    'es': TranslationTemplate(
        lang_code='es',
        belief_prefix="[CREENCIA]",
        confidence_label="Confianza",
        updated_label="Actualizado",
        sources_label="Fuentes",
        context_label="Contexto",
        header="# Creencias Relevantes",
        tension_header="# ⚠️ Tensiones (Contradicciones)"
    ),

    'fr': TranslationTemplate(
        lang_code='fr',
        belief_prefix="[CROYANCE]",
        confidence_label="Confiance",
        updated_label="Mis à jour",
        sources_label="Sources",
        context_label="Contexte",
        header="# Croyances Pertinentes",
        tension_header="# ⚠️ Tensions (Contradictions)"
    ),

    'de': TranslationTemplate(
        lang_code='de',
        belief_prefix="[ÜBERZEUGUNG]",
        confidence_label="Vertrauen",
        updated_label="Aktualisiert",
        sources_label="Quellen",
        context_label="Kontext",
        header="# Relevante Überzeugungen",
        tension_header="# ⚠️ Spannungen (Widersprüche)"
    ),
}


# ============================================================================
# Card Translator
# ============================================================================

class CardTranslator:
    """
    Translates belief cards to target language.

    Note: This uses simple template-based translation for labels.
    For full content translation, integrate with a translation service
    (e.g., Google Translate API, DeepL, etc.)
    """

    def __init__(self,
                 templates: Optional[Dict[str, TranslationTemplate]] = None,
                 translate_content: bool = False):
        """
        Initialize translator.

        Args:
            templates: Custom translation templates
            translate_content: Whether to translate belief content
                              (requires translation service integration)
        """
        self.templates = templates or TRANSLATION_TEMPLATES.copy()
        self.translate_content = translate_content

    def get_template(self, lang_code: str) -> TranslationTemplate:
        """
        Get translation template for language.

        Args:
            lang_code: Language code

        Returns:
            Translation template
        """
        return self.templates.get(lang_code, TRANSLATION_TEMPLATES['en'])

    def translate_card(self, card_text: str, target_lang: str) -> str:
        """
        Translate belief card to target language.

        Translates labels/structure but keeps sources in original.

        Args:
            card_text: Original card text
            target_lang: Target language code

        Returns:
            Translated card
        """
        if target_lang == 'en':
            return card_text  # Already in English

        template = self.get_template(target_lang)
        en_template = self.get_template('en')

        # Replace labels
        translated = card_text

        # Replace belief prefix
        translated = translated.replace(
            en_template.belief_prefix,
            template.belief_prefix
        )

        # Replace labels
        replacements = {
            f"{en_template.confidence_label}:": f"{template.confidence_label}:",
            f"{en_template.updated_label}:": f"{template.updated_label}:",
            f"{en_template.sources_label}:": f"{template.sources_label}:",
            f"{en_template.context_label}:": f"{template.context_label}:",
        }

        for en_label, target_label in replacements.items():
            translated = translated.replace(en_label, target_label)

        # Note: Content translation would happen here if enabled
        # if self.translate_content:
        #     translated_content = translate_text(content, target_lang)

        return translated

    def translate_context_pack(self, context_pack: str, target_lang: str) -> str:
        """
        Translate entire context pack.

        Args:
            context_pack: Original context pack
            target_lang: Target language

        Returns:
            Translated pack
        """
        if target_lang == 'en':
            return context_pack

        template = self.get_template(target_lang)
        en_template = self.get_template('en')

        # Translate headers
        translated = context_pack.replace(
            en_template.header,
            template.header
        )

        translated = translated.replace(
            en_template.tension_header,
            template.tension_header
        )

        # Translate individual cards
        lines = translated.split('\n')
        translated_lines = []

        for line in lines:
            if line.strip().startswith(en_template.belief_prefix):
                # Translate this card line
                translated_line = self.translate_card(line, target_lang)
                translated_lines.append(translated_line)
            elif any(label in line for label in [
                en_template.confidence_label,
                en_template.updated_label,
                en_template.sources_label,
                en_template.context_label
            ]):
                # Translate label line
                translated_line = self.translate_card(line, target_lang)
                translated_lines.append(translated_line)
            else:
                # Keep line as-is (sources, tension content, etc.)
                translated_lines.append(line)

        return '\n'.join(translated_lines)


# ============================================================================
# Content Translation (Placeholder for Real Service)
# ============================================================================

def translate_text(text: str, target_lang: str, source_lang: str = 'en') -> str:
    """
    Translate text content.

    PLACEHOLDER: Integrate with real translation service (Google Translate, DeepL, etc.)

    Args:
        text: Text to translate
        target_lang: Target language code
        source_lang: Source language code

    Returns:
        Translated text (currently returns original)
    """
    # TODO: Integrate with translation API
    # Example with Google Translate:
    # from googletrans import Translator
    # translator = Translator()
    # result = translator.translate(text, src=source_lang, dest=target_lang)
    # return result.text

    # For now, return original
    return text


# ============================================================================
# Convenience Functions
# ============================================================================

def auto_translate_card(card_text: str, prompt: str) -> str:
    """
    Auto-translate card based on prompt language.

    Args:
        card_text: Original card
        prompt: User prompt (for language detection)

    Returns:
        Translated card
    """
    target_lang = detect_language(prompt)
    translator = CardTranslator()
    return translator.translate_card(card_text, target_lang)


def auto_translate_context_pack(context_pack: str, prompt: str) -> str:
    """
    Auto-translate context pack based on prompt language.

    Args:
        context_pack: Original pack
        prompt: User prompt

    Returns:
        Translated pack
    """
    target_lang = detect_language(prompt)
    translator = CardTranslator()
    return translator.translate_context_pack(context_pack, target_lang)
