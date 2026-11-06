"""
Neura Multilingual Smart Router
================================

Deterministic multilingual intent routing layer.

This layer is intentionally independent from the LLM.
It provides useful responses for common intents and routes
URLs/tool requests before falling back to the language model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class SmartRoute:
    action: str
    language: str
    response: str = ""
    url: str = ""
    confidence: float = 0.0


class MultilingualSmartRouter:
    """Lightweight multilingual intent router."""

    LANGUAGE_PATTERNS = {
        "ar": re.compile(r"[\u0600-\u06FF]"),
        "fa": re.compile(r"[\u067E\u0686\u0698\u06AF\u06CC]"),
        "ur": re.compile(r"[\u0679\u0688\u06D2\u06BA\u06BE\u06C1\u06CC]"),
        "ru": re.compile(r"[\u0400-\u04FF]"),
        "el": re.compile(r"[\u0370-\u03FF]"),
        "he": re.compile(r"[\u0590-\u05FF]"),
        "zh": re.compile(r"[\u4E00-\u9FFF]"),
        "ja": re.compile(r"[\u3040-\u30FF]"),
        "ko": re.compile(r"[\uAC00-\uD7AF]"),
        "th": re.compile(r"[\u0E00-\u0E7F]"),
        "hi": re.compile(r"[\u0900-\u097F]"),
        "bn": re.compile(r"[\u0980-\u09FF]"),
        "ta": re.compile(r"[\u0B80-\u0BFF]"),
        "te": re.compile(r"[\u0C00-\u0C7F]"),
        "ka": re.compile(r"[\u10A0-\u10FF]"),
        "hy": re.compile(r"[\u0530-\u058F]"),
    }

    GREETINGS = {
        "ar": (
            "السلام عليكم",
            "وعليكم السلام ورحمة الله وبركاته 🌹\nكيف يمكنني مساعدتك؟",
        ),
        "en": (
            "hello",
            "Hello! 👋 How can I help you?",
        ),
        "fr": (
            "bonjour",
            "Bonjour ! 👋 Comment puis-je vous aider ?",
        ),
        "de": (
            "hallo",
            "Hallo! 👋 Wie kann ich dir helfen?",
        ),
        "es": (
            "hola",
            "¡Hola! 👋 ¿Cómo puedo ayudarte?",
        ),
        "it": (
            "ciao",
            "Ciao! 👋 Come posso aiutarti?",
        ),
        "tr": (
            "merhaba",
            "Merhaba! 👋 Size nasıl yardımcı olabilirim?",
        ),
        "fa": (
            "سلام",
            "سلام! 👋 چطور می‌توانم کمکتان کنم؟",
        ),
        "ru": (
            "привет",
            "Привет! 👋 Чем я могу помочь?",
        ),
        "zh": (
            "你好",
            "你好！👋 我能帮助你什么？",
        ),
        "ja": (
            "こんにちは",
            "こんにちは！👋 何をお手伝いできますか？",
        ),
        "ko": (
            "안녕하세요",
            "안녕하세요! 👋 무엇을 도와드릴까요?",
        ),
        "hi": (
            "नमस्ते",
            "नमस्ते! 👋 मैं आपकी कैसे मदद कर सकता हूँ?",
        ),
        "ur": (
            "السلام علیکم",
            "وعلیکم السلام! 👋 میں آپ کی کیسے مدد کر سکتا ہوں؟",
        ),
    }

    HOW_ARE_YOU = {
        "ar": [
            "كيف حالك",
            "شلونك",
            "شخبارك",
            "كيفك",
        ],
        "en": [
            "how are you",
            "how are u",
        ],
        "fr": [
            "comment allez-vous",
            "comment ça va",
        ],
        "de": [
            "wie geht es dir",
            "wie geht's dir",
        ],
        "es": [
            "como estas",
            "cómo estás",
        ],
        "it": [
            "come stai",
        ],
        "tr": [
            "nasılsın",
            "nasilsin",
        ],
        "fa": [
            "حالت چطوره",
            "حال شما چطور است",
        ],
        "ru": [
            "как дела",
        ],
        "zh": [
            "你好吗",
            "最近怎么样",
        ],
        "ja": [
            "元気ですか",
        ],
        "ko": [
            "잘 지내세요",
        ],
        "hi": [
            "आप कैसे हैं",
        ],
        "ur": [
            "آپ کیسے ہیں",
        ],
    }

    IDENTITY = {
        "ar": ["من انت", "من أنت", "ما اسمك", "شنو اسمك"],
        "en": ["who are you", "what is your name"],
        "fr": ["qui es-tu", "comment tu t'appelles"],
        "de": ["wer bist du", "wie heißt du"],
        "es": ["quién eres", "como te llamas"],
        "it": ["chi sei", "come ti chiami"],
        "tr": ["sen kimsin", "adın ne"],
        "fa": ["تو کی هستی", "اسمت چیست"],
        "ru": ["кто ты", "как тебя зовут"],
        "zh": ["你是谁", "你叫什么名字"],
        "ja": ["あなたは誰", "名前は何"],
        "ko": ["누구세요", "이름이 뭐예요"],
        "hi": ["आप कौन हैं", "आपका नाम क्या है"],
    }

    PROGRAMMING = [
        "python",
        "javascript",
        "typescript",
        "java",
        "c++",
        "c#",
        "golang",
        "rust",
        "kotlin",
        "swift",
        "php",
        "ruby",
        "dart",
        "sql",
        "html",
        "css",
        "bash",
        "shell",
        "programming",
        "programmer",
        "coding",
        "code",
        "برمجة",
        "مبرمج",
        "كود",
        "شفرة",
        "بايثون",
        "جافاسكربت",
        "سي++",
        "جافا",
        "روست",
        "برمجة تطبيقات",
        "برمجة مواقع",
        "كيف ابرمج",
        "كيف أبرمج",
        "شلون ابرمج",
        "شلون أبرمج",
        "تعلم البرمجة",
        "تعلم برمجة",
        "اريد اتعلم البرمجة",
        "أريد أن أتعلم البرمجة",
        "كيف اكتب كود",
        "كيف أكتب كود",
        "كيف اكتب برنامج",
        "كيف أكتب برنامج",
    ]

    ARABIC_KNOWLEDGE = {
        "عاصمة العراق": "عاصمة العراق هي بغداد.",
        "ما هي عاصمة العراق": "عاصمة العراق هي بغداد.",
        "ماهي عاصمة العراق": "عاصمة العراق هي بغداد.",
        "ماهيه عاصمه العراق": "عاصمة العراق هي بغداد.",
        "عاصمه العراق": "عاصمة العراق هي بغداد.",
    }

    URL_RE = re.compile(
        r"(?i)\bhttps?://[^\s<>\"]+"
    )

    DOMAIN_RE = re.compile(
        r"(?i)\b(?:www\.)?"
        r"(?:github\.com|google\.com|youtube\.com|wikipedia\.org|"
        r"openai\.com|python\.org|pytorch\.org|huggingface\.co|"
        r"stackoverflow\.com|stackoverflow\.com)"
        r"(?:/[^\s<>\"]*)?"
    )

    def detect_language(self, text: str) -> str:
        """Best-effort language detection without external dependencies."""

        text = str(text or "").strip()

        if not text:
            return "unknown"

        scores = {
            language: len(pattern.findall(text))
            for language, pattern in self.LANGUAGE_PATTERNS.items()
        }

        best_script = max(scores, key=scores.get)

        if scores[best_script] > 0:
            if best_script == "fa":
                return "fa"
            if best_script == "ur":
                return "ur"
            return best_script

        lowered = text.lower()

        latin_markers = {
            "fr": (
                "bonjour", "merci", "comment", "pourquoi",
                "avec", "être", "est",
            ),
            "de": (
                "hallo", "danke", "wie", "warum",
                "und", "nicht", "ist",
            ),
            "es": (
                "hola", "gracias", "como", "qué",
                "porque", "para", "está",
            ),
            "it": (
                "ciao", "grazie", "come", "perché",
                "questo", "sono",
            ),
            "tr": (
                "merhaba", "nasılsın", "nasilsin",
                "teşekkür", "tesekkur", "için", "icin",
            ),
        }

        for language, words in latin_markers.items():
            if any(word in lowered for word in words):
                return language

        return "en"

    def _clean(self, text: str) -> str:
        text = str(text or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    def extract_url(self, text: str) -> str | None:
        match = self.URL_RE.search(text)

        if match:
            return match.group(0).rstrip(".,!?،؛")

        match = self.DOMAIN_RE.search(text)

        if match:
            value = match.group(0).rstrip(".,!?،؛")
            if not value.startswith(("http://", "https://")):
                value = "https://" + value
            return value

        return None

    def _is_safe_url(self, url: str) -> bool:
        try:
            parsed = urlparse(url)

            if parsed.scheme not in ("http", "https"):
                return False

            host = (parsed.hostname or "").lower()

            if not host:
                return False

            blocked_hosts = {
                "localhost",
                "localhost.localdomain",
                "0.0.0.0",
                "::1",
            }

            if host in blocked_hosts:
                return False

            if host.endswith(".local"):
                return False

            return True

        except Exception:
            return False

    def _contains_any(self, text: str, values) -> bool:
        return any(value in text for value in values)

    def route(self, text: str) -> SmartRoute:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        original = text.strip()
        lowered = self._clean(original)
        language = self.detect_language(original)

        if not lowered:
            return SmartRoute(
                action="empty",
                language=language,
                confidence=1.0,
            )

        # URLs are detected early, but known tool URLs get
        # their own routing action.
        url = self.extract_url(original)

        if url:
            parsed_host = (urlparse(url).hostname or "").lower()

            if parsed_host in {"github.com", "www.github.com"}:
                return SmartRoute(
                    action="github",
                    language=language,
                    url=url,
                    confidence=1.0,
                )

            if not self._is_safe_url(url):
                return SmartRoute(
                    action="unsafe_url",
                    language=language,
                    response=(
                        "لا أستطيع فتح هذا الرابط لأسباب أمنية."
                        if language == "ar"
                        else "I cannot open this URL for security reasons."
                    ),
                    url=url,
                    confidence=1.0,
                )

            return SmartRoute(
                action="url",
                language=language,
                url=url,
                confidence=0.99,
            )

        # Greetings.
        for lang, (trigger, response) in self.GREETINGS.items():
            if lowered == trigger:
                return SmartRoute(
                    action="greeting",
                    language=lang,
                    response=response,
                    confidence=0.99,
                )

        # How are you?
        for lang, triggers in self.HOW_ARE_YOU.items():
            if self._contains_any(lowered, triggers):
                responses = {
                    "ar": "بخير والحمد لله 🌹 وأنا جاهز لمساعدتك.",
                    "en": "I'm doing well and ready to help you.",
                    "fr": "Je vais bien et je suis prêt à vous aider.",
                    "de": "Mir geht es gut und ich bin bereit zu helfen.",
                    "es": "Estoy bien y listo para ayudarte.",
                    "it": "Sto bene e sono pronto ad aiutarti.",
                    "tr": "İyiyim ve size yardımcı olmaya hazırım.",
                    "fa": "خوبم و آماده‌ام به شما کمک کنم.",
                    "ru": "У меня всё хорошо, и я готов помочь.",
                    "zh": "我很好，随时可以帮助你。",
                    "ja": "元気です。いつでもお手伝いできます。",
                    "ko": "잘 지내고 있어요. 도와드릴 준비가 되어 있습니다.",
                    "hi": "मैं ठीक हूँ और आपकी मदद करने के लिए तैयार हूँ।",
                    "ur": "میں ٹھیک ہوں اور آپ کی مدد کے لیے تیار ہوں۔",
                }

                return SmartRoute(
                    action="smalltalk",
                    language=lang,
                    response=responses.get(
                        lang,
                        responses["en"],
                    ),
                    confidence=0.98,
                )

        # Identity.
        for lang, triggers in self.IDENTITY.items():
            if self._contains_any(lowered, triggers):
                responses = {
                    "ar": "أنا Neura، مساعد ذكاء اصطناعي متعدد اللغات، وما زلت في مرحلة التطوير والتحسين المستمر.",
                    "en": "I am Neura, a multilingual AI assistant under continuous development and improvement.",
                    "fr": "Je suis Neura, un assistant IA multilingue en développement continu.",
                    "de": "Ich bin Neura, ein mehrsprachiger KI-Assistent in kontinuierlicher Entwicklung.",
                    "es": "Soy Neura, un asistente de IA multilingüe en desarrollo continuo.",
                    "it": "Sono Neura, un assistente IA multilingue in continuo sviluppo.",
                    "tr": "Ben Neura'yım; sürekli geliştirilen çok dilli bir yapay zekâ asistanıyım.",
                    "fa": "من Neura هستم؛ یک دستیار هوش مصنوعی چندزبانه که دائماً در حال توسعه است.",
                    "ru": "Я Neura — многоязычный ИИ-ассистент, который постоянно развивается.",
                    "zh": "我是 Neura，一个持续开发和改进中的多语言 AI 助手。",
                    "ja": "私はNeuraです。継続的に開発・改善されている多言語AIアシスタントです。",
                    "ko": "저는 Neura입니다. 계속 개발되고 개선되는 다국어 AI 어시스턴트입니다.",
                    "hi": "मैं Neura हूँ, एक बहुभाषी AI सहायक जिसे लगातार विकसित किया जा रहा है।",
                }

                return SmartRoute(
                    action="identity",
                    language=lang,
                    response=responses.get(lang, responses["en"]),
                    confidence=0.98,
                )

        # Stable Arabic knowledge examples.
        for key, response in self.ARABIC_KNOWLEDGE.items():
            if key in lowered:
                return SmartRoute(
                    action="knowledge",
                    language="ar",
                    response=response,
                    confidence=0.99,
                )

        # Learning/resource requests must be handled before programming.
        # This allows queries such as "I want to learn Python" to reach
        # the curated knowledge-source pipeline instead of the generic
        # programming response.
        learning_triggers = (
            "learn",
            "learning",
            "tutorial",
            "tutorials",
            "course",
            "courses",
            "study",
            "how to learn",
            "resources to learn",
            "تعلم",
            "تعلّم",
            "اتعلم",
            "أتعلم",
            "دروس",
            "دورة",
            "دورات",
            "كورس",
            "كورسات",
            "شرح",
            "دليل",
        )

        learning_topics = (
            "python",
            "pytorch",
            "hugging face",
            "huggingface",
            "javascript",
            "typescript",
            "html",
            "css",
            "web development",
            "web development",
            "machine learning",
            "deep learning",
            "artificial intelligence",
            "ai",
            "llm",
            "transformer",
            "github",
            "rust",
            "docker",
            "numpy",
            "scipy",
            "kaggle",
            "arxiv",
            "data science",
            "programming",
            "برمجة",
            "بايثون",
            "بايتورتش",
            "ذكاء اصطناعي",
            "الذكاء الاصطناعي",
            "تعلم آلي",
            "تعلم عميق",
            "تطوير الويب",
            "تطوير المواقع",
            "نماذج لغوية",
            "محولات",
            "علم البيانات",
            "بحث علمي",
        )

        if (
            self._contains_any(lowered, learning_triggers)
            and self._contains_any(lowered, learning_topics)
        ):
            return SmartRoute(
                action="knowledge_source",
                language=language,
                response="",
                confidence=0.97,
            )

        # Programming requests.
        if self._contains_any(lowered, self.PROGRAMMING):
            responses = {
                "ar": (
                    "بالتأكيد. أستطيع مساعدتك في البرمجة، "
                    "مثل Python وJavaScript وC/C++ وJava وRust وGo وSQL وغيرها. "
                    "أرسل لي الكود أو اشرح المطلوب وسأساعدك خطوة بخطوة."
                ),
                "en": (
                    "Absolutely. I can help with programming, including Python, "
                    "JavaScript, C/C++, Java, Rust, Go, SQL and more. "
                    "Send me your code or describe what you want to build."
                ),
                "fr": (
                    "Bien sûr. Je peux vous aider en programmation, notamment "
                    "Python, JavaScript, C/C++, Java, Rust, Go, SQL et plus."
                ),
                "de": (
                    "Natürlich. Ich kann bei Python, JavaScript, C/C++, Java, "
                    "Rust, Go, SQL und vielen weiteren Sprachen helfen."
                ),
                "es": (
                    "Claro. Puedo ayudarte con Python, JavaScript, C/C++, Java, "
                    "Rust, Go, SQL y muchos otros lenguajes."
                ),
            }

            return SmartRoute(
                action="programming",
                language=language,
                response=responses.get(language, responses["en"]),
                confidence=0.94,
            )

        return SmartRoute(
            action="model",
            language=language,
            confidence=0.40,
        )


SMART_ROUTER = MultilingualSmartRouter()
