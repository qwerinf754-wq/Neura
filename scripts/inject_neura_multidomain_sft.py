from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("data/sft")
ROOT.mkdir(parents=True, exist_ok=True)


def msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def ex(
    domain: str,
    category: str,
    user: str,
    assistant: str,
    language: str = "ar",
    system: str = (
        "أنت Neura، مساعد ذكاء اصطناعي متعدد المجالات. "
        "كن دقيقًا ومفيدًا وواضحًا، وإذا كان الطلب يحتاج أداة خارجية "
        "فاشرح ما يمكن تنفيذه وما يحتاج أداة متخصصة."
    ),
):
    return {
        "metadata": {
            "domain": domain,
            "category": category,
            "language": language,
        },
        "messages": [
            msg("system", system),
            msg("user", user),
            msg("assistant", assistant),
        ],
    }


DATA: dict[str, list[dict]] = {}


# ============================================================
# 1. ARABIC
# ============================================================

DATA["neura_arabic_world.jsonl"] = [
    ex(
        "arabic",
        "grammar",
        "أعرب الجملة: العلم نور.",
        "العلم: مبتدأ مرفوع وعلامة رفعه الضمة. نور: خبر مرفوع وعلامة رفعه الضمة.",
    ),
    ex(
        "arabic",
        "writing",
        "اكتب فقرة عربية فصيحة عن أهمية العلم.",
        "العلم أساس تقدم الأمم، فهو يفتح أبواب المعرفة ويمنح الإنسان القدرة على فهم العالم وحل المشكلات. وبالعلم تتطور المجتمعات وتتحسن حياة الناس.",
    ),
    ex(
        "arabic",
        "summarization",
        "لخّص فكرة نص طويل في نقاط.",
        "أرسل النص، وسأستخرج الفكرة الرئيسية ثم أهم النقاط والنتائج بصورة مختصرة ومنظمة.",
    ),
    ex(
        "arabic",
        "spelling",
        "صحح الجملة: هاذا كتاب جميل جداً.",
        "الصحيح: هذا كتاب جميل جدًا.",
    ),
    ex(
        "arabic",
        "formal",
        "حوّل هذه العبارة إلى أسلوب رسمي: أريد أعرف شنو صار.",
        "أرغب في معرفة ما الذي حدث.",
    ),
    ex(
        "arabic",
        "explanation",
        "اشرح الفرق بين اللغة الفصحى واللهجات العربية.",
        "العربية الفصحى هي الصيغة المعيارية المستخدمة في التعليم والإعلام والكتابة الرسمية، بينما اللهجات العربية هي تنوعات محلية تطورت في مناطق مختلفة وتختلف في النطق والمفردات وبعض التراكيب.",
    ),
]


# ============================================================
# 2. IRAQI
# ============================================================

DATA["neura_iraqi.jsonl"] = [
    ex(
        "iraqi",
        "conversation",
        "شلونك Neura؟",
        "تمام والحمد لله 🌹 شلونك إنت؟ شنو أگدر أساعدك بيه اليوم؟",
        language="ar-IQ",
    ),
    ex(
        "iraqi",
        "explanation",
        "أريد شرح بسيط للذكاء الاصطناعي باللهجة العراقية.",
        "بشكل بسيط، الذكاء الاصطناعي هو أن نخلي الكمبيوتر يتعلم من البيانات حتى يگدر يفهم أنماط ويساعدنا بمهام مثل الكتابة والترجمة والبرمجة وتحليل المعلومات.",
        language="ar-IQ",
    ),
    ex(
        "iraqi",
        "planning",
        "عندي فكرة مشروع بالعراق، شلون أبدأ؟",
        "نبدأ بتحديد المشكلة اللي المشروع يحلها، وبعدها الزبون المستهدف، والتكلفة، والسعر، والمنافسين، وطريقة التسويق. بعدها نسوي نسخة صغيرة نجربها بالسوق قبل ما نصرف مبلغ كبير.",
        language="ar-IQ",
    ),
    ex(
        "iraqi",
        "dialogue",
        "ما أعرف أبرمج، بس أريد أسوي تطبيق.",
        "ماكو مشكلة. نحدد أولًا فكرة التطبيق والوظائف المطلوبة، وبعدها أگدر أساعدك بتصميم الواجهة، قاعدة البيانات، الـAPI والكود خطوة بخطوة.",
        language="ar-IQ",
    ),
    ex(
        "iraqi",
        "translation",
        "ترجم للعراقي: I don't understand this error.",
        "ما دا أفهم هذا الخطأ.",
        language="ar-IQ",
    ),
]


# ============================================================
# 3. WORLD LANGUAGES / TRANSLATION
# ============================================================

DATA["neura_translation.jsonl"] = [
    ex(
        "translation",
        "english_arabic",
        "ترجم إلى العربية: Artificial intelligence can help programmers.",
        "يمكن للذكاء الاصطناعي أن يساعد المبرمجين.",
        language="en-ar",
    ),
    ex(
        "translation",
        "arabic_english",
        "ترجم إلى الإنجليزية: أريد بناء نموذج ذكاء اصطناعي.",
        "I want to build an artificial intelligence model.",
        language="ar-en",
    ),
    ex(
        "translation",
        "arabic_french",
        "ترجم إلى الفرنسية: مرحبًا، كيف حالك؟",
        "Bonjour, comment allez-vous ?",
        language="ar-fr",
    ),
    ex(
        "translation",
        "arabic_spanish",
        "ترجم إلى الإسبانية: أنا أتعلم البرمجة.",
        "Estoy aprendiendo programación.",
        language="ar-es",
    ),
    ex(
        "translation",
        "arabic_german",
        "ترجم إلى الألمانية: أريد تعلم لغة جديدة.",
        "Ich möchte eine neue Sprache lernen.",
        language="ar-de",
    ),
    ex(
        "translation",
        "arabic_turkish",
        "ترجم إلى التركية: أين محطة القطار؟",
        "Tren istasyonu nerede?",
        language="ar-tr",
    ),
    ex(
        "translation",
        "arabic_persian",
        "ترجم إلى الفارسية: شكرًا لك.",
        "ممنونم.",
        language="ar-fa",
    ),
    ex(
        "translation",
        "arabic_russian",
        "ترجم إلى الروسية: صباح الخير.",
        "Доброе утро.",
        language="ar-ru",
    ),
    ex(
        "translation",
        "arabic_japanese",
        "ترجم إلى اليابانية: مرحبًا.",
        "こんにちは。",
        language="ar-ja",
    ),
    ex(
        "translation",
        "arabic_chinese",
        "ترجم إلى الصينية: أنا أحب التعلم.",
        "我喜欢学习。",
        language="ar-zh",
    ),
    ex(
        "translation",
        "arabic_korean",
        "ترجم إلى الكورية: كيف حالك؟",
        "잘 지내세요?",
        language="ar-ko",
    ),
    ex(
        "translation",
        "translation_quality",
        "هل الترجمة الحرفية أفضل دائمًا؟",
        "لا. الترجمة الجيدة تحافظ على المعنى والسياق والأسلوب، وقد تحتاج إلى إعادة صياغة بدل ترجمة كل كلمة حرفيًا.",
    ),
]


# ============================================================
# 4. PROGRAMMING
# ============================================================

DATA["neura_programming_world.jsonl"] = [
    ex(
        "programming",
        "python",
        "اكتب دالة Python تحسب متوسط قائمة أرقام.",
        """```python
def average(numbers):
    if not numbers:
        raise ValueError("numbers cannot be empty")
    return sum(numbers) / len(numbers)
```""",
    ),
    ex(
        "programming",
        "javascript",
        "اكتب JavaScript يحول قائمة أسماء إلى أحرف كبيرة.",
        """```javascript
const names = ["ali", "neura"];
const result = names.map(name => name.toUpperCase());
```""",
    ),
    ex(
        "programming",
        "typescript",
        "اكتب TypeScript لنوع User.",
        """```typescript
interface User {
    id: number;
    name: string;
    email?: string;
}
```""",
    ),
    ex(
        "programming",
        "cpp",
        "اكتب C++ لحساب مجموع عناصر vector.",
        """```cpp
#include <numeric>
#include <vector>

int total(const std::vector<int>& values) {
    return std::accumulate(values.begin(), values.end(), 0);
}
```""",
    ),
    ex(
        "programming",
        "java",
        "كيف تنشئ قائمة أعداد في Java؟",
        """```java
List<Integer> numbers = new ArrayList<>();
numbers.add(10);
numbers.add(20);
```""",
    ),
    ex(
        "programming",
        "csharp",
        "اكتب C# لتحويل قائمة أرقام إلى مربعاتها.",
        """```csharp
var squares = numbers.Select(x => x * x).ToList();
```""",
    ),
    ex(
        "programming",
        "go",
        "اكتب دالة Go تجمع رقمين.",
        """```go
func Add(a int, b int) int {
    return a + b
}
```""",
    ),
    ex(
        "programming",
        "rust",
        "اكتب Rust لدالة تضرب رقمين.",
        """```rust
fn multiply(a: i32, b: i32) -> i32 {
    a * b
}
```""",
    ),
    ex(
        "programming",
        "php",
        "اكتب PHP يطبع اسم المستخدم.",
        """```php
<?php

$name = "Neura";
echo $name;
```""",
    ),
    ex(
        "programming",
        "sql",
        "اكتب SQL يعرض المستخدمين مرتبين حسب الاسم.",
        """```sql
SELECT id, name
FROM users
ORDER BY name ASC;
```""",
    ),
    ex(
        "programming",
        "bash",
        "اكتب Bash ينشئ مجلدًا إذا لم يكن موجودًا.",
        """```bash
mkdir -p data/output
```""",
    ),
    ex(
        "programming",
        "debugging",
        "كيف تشخص خطأ برمجي؟",
        "أبدأ برسالة الخطأ وStack Trace، ثم أحدد السطر المسؤول، وأعيد إنتاج المشكلة، وأفحص المدخلات والحالة، وبعدها أختبر الإصلاح بحالة اختبار واضحة.",
    ),
]


# ============================================================
# 5. MATH
# ============================================================

DATA["neura_math.jsonl"] = [
    ex(
        "math",
        "arithmetic",
        "احسب 25 × 16.",
        "25 × 16 = 400.",
    ),
    ex(
        "math",
        "algebra",
        "حل المعادلة 2x + 6 = 16.",
        "نطرح 6 من الطرفين: 2x = 10. ثم نقسم على 2: x = 5.",
    ),
    ex(
        "math",
        "quadratic",
        "حل x² - 5x + 6 = 0.",
        "نحللها إلى (x - 2)(x - 3) = 0، لذلك الحلول هي x = 2 وx = 3.",
    ),
    ex(
        "math",
        "percentage",
        "ما قيمة 15% من 200؟",
        "15% من 200 تساوي 30.",
    ),
    ex(
        "math",
        "geometry",
        "ما مساحة مستطيل طوله 8 وعرضه 5؟",
        "المساحة = الطول × العرض = 8 × 5 = 40 وحدة مربعة.",
    ),
    ex(
        "math",
        "probability",
        "ما احتمال ظهور صورة عند رمي قطعة نقد عادلة؟",
        "الاحتمال هو 1/2، أي 50%.",
    ),
    ex(
        "math",
        "word_problem",
        "سيارة قطعت 240 كيلومترًا خلال 4 ساعات. ما متوسط السرعة؟",
        "متوسط السرعة = المسافة ÷ الزمن = 240 ÷ 4 = 60 كم/ساعة.",
    ),
    ex(
        "math",
        "reasoning",
        "إذا كان لدى أحمد 20 كتابًا وأعطى 7 كتب، كم بقي لديه؟",
        "20 - 7 = 13 كتابًا.",
    ),
]


# ============================================================
# 6. GENERAL KNOWLEDGE / EXPERT
# ============================================================

DATA["neura_knowledge.jsonl"] = [
    ex(
        "knowledge",
        "science",
        "ما هي عملية البناء الضوئي؟",
        "هي العملية التي تستخدم فيها النباتات الضوء لتحويل الماء وثاني أكسيد الكربون إلى مواد عضوية مثل الجلوكوز، مع إطلاق الأكسجين.",
    ),
    ex(
        "knowledge",
        "computer",
        "ما الفرق بين RAM والتخزين؟",
        "RAM ذاكرة سريعة مؤقتة تستخدم أثناء تشغيل البرامج، بينما التخزين مثل SSD يحتفظ بالبيانات حتى بعد إيقاف تشغيل الجهاز.",
    ),
    ex(
        "knowledge",
        "network",
        "ما وظيفة DNS؟",
        "DNS يحول أسماء النطاقات مثل example.com إلى عناوين IP يمكن للأجهزة استخدامها للوصول إلى الخوادم.",
    ),
    ex(
        "knowledge",
        "ai",
        "ما هو Transformer في الذكاء الاصطناعي؟",
        "هو نوع من بنى الشبكات العصبية يعتمد على آلية attention لمعالجة العلاقات بين عناصر التسلسل، وأصبح أساسًا لعدد كبير من نماذج اللغة الحديثة.",
    ),
    ex(
        "knowledge",
        "security",
        "ما معنى التشفير؟",
        "هو تحويل البيانات إلى صيغة غير مفهومة دون مفتاح أو طريقة فك مناسبة، بهدف حماية سرية المعلومات.",
    ),
    ex(
        "knowledge",
        "learning",
        "كيف أتعلم موضوعًا تقنيًا بسرعة؟",
        "حدد المفاهيم الأساسية أولًا، ثم طبّقها بمشاريع صغيرة، واختبر فهمك بحل مسائل، وراجع الأخطاء بدل الاكتفاء بالقراءة النظرية.",
    ),
]


# ============================================================
# 7. IMAGE DESIGN
# ============================================================

DATA["neura_image_design.jsonl"] = [
    ex(
        "image_design",
        "prompt",
        "أريد Prompt لصورة سينمائية لمدينة مستقبلية ليلًا.",
        """Prompt مقترح:

"مشهد سينمائي لمدينة مستقبلية ضخمة ليلًا، ناطحات سحاب مضيئة، شوارع مبللة تعكس أضواء النيون، سيارات مستقبلية، ضباب خفيف، إضاءة درامية، تفاصيل عالية، عمق ميدان سينمائي، تكوين واسع، واقعية فوتوغرافية، أجواء مستقبلية غامضة."

يمكن تخصيص الألوان والأسلوب ونسبة الأبعاد حسب الأداة المستخدمة.""",
    ),
    ex(
        "image_design",
        "portrait",
        "صمم وصفًا لصورة شخصية احترافية.",
        "صورة بورتريه احترافية، إضاءة استوديو ناعمة، خلفية بسيطة، تكوين متوازن، تفاصيل طبيعية للوجه، عمق ميدان خفيف، مظهر واقعي، جودة فوتوغرافية عالية.",
    ),
    ex(
        "image_design",
        "product",
        "أريد صورة إعلانية لمنتج تقني.",
        "لقطة منتج تقنية احترافية على سطح نظيف، إضاءة استوديو، انعكاسات محسوبة، خلفية حديثة، تكوين تجاري أنيق، تركيز واضح على المنتج ومساحة مناسبة للنص الإعلاني.",
    ),
    ex(
        "image_design",
        "storyboard",
        "كيف تصمم صورة لمشهد قصة؟",
        "أحدد الشخصيات والمكان والوقت والإضاءة والمزاج وزاوية الكاميرا والعناصر المهمة، ثم أحولها إلى Prompt متماسك يحافظ على استمرارية المشهد.",
    ),
]


# ============================================================
# 8. VIDEO DESIGN
# ============================================================

DATA["neura_video_design.jsonl"] = [
    ex(
        "video_design",
        "cinematic",
        "اكتب Prompt لفيديو سينمائي لسيارة تسير ليلًا.",
        """مشهد فيديو سينمائي لسيارة رياضية تسير في شارع مدينة ليلية مبللة بالمطر، انعكاسات أضواء النيون على الأسفلت، حركة كاميرا tracking جانبية ناعمة، عمق ميدان سينمائي، ضباب خفيف، إضاءة واقعية، حركة طبيعية، مدة 8 ثوانٍ.""",
    ),
    ex(
        "video_design",
        "camera",
        "ما المعلومات المهمة عند كتابة Prompt لفيديو؟",
        "الموضوع، المكان، الحركة، زاوية الكاميرا، نوع حركة الكاميرا، الإضاءة، الزمن، الأسلوب البصري، سرعة الحركة، مدة المشهد ونسبة الأبعاد.",
    ),
    ex(
        "video_design",
        "advertising",
        "صمم فكرة فيديو إعلان قصير لمنتج تقني.",
        "يبدأ الفيديو بلقطة قريبة للمنتج، ثم تتحرك الكاميرا للخلف لتظهر البيئة، تظهر ثلاث فوائد رئيسية عبر انتقالات بسيطة، ثم تنتهي اللقطة بالمنتج والشعار ونداء واضح لاتخاذ الإجراء.",
    ),
]


# ============================================================
# 9. MUSIC / SONGWRITING
# ============================================================

DATA["neura_music.jsonl"] = [
    ex(
        "music",
        "songwriting",
        "اكتب فكرة أغنية عربية عن الأمل.",
        "الفكرة: شخص مر بمرحلة صعبة لكنه يقرر الاستمرار. المقطع الأول يصف التعب، ثم يأتي الـPre-Chorus بتصاعد تدريجي، ويكون الـChorus عن النهوض وبداية جديدة.",
    ),
    ex(
        "music",
        "lyrics",
        "اكتب كلمات أصلية قصيرة لأغنية عن الحلم.",
        "أمشي وطريقي ما انتهى\nوالحلم قدامي ما انطفى\nكل ما تعبت أبدأ من جديد\nوأشوف بكرة أقرب من بعيد",
    ),
    ex(
        "music",
        "composition",
        "كيف أصمم أغنية من الصفر؟",
        "ابدأ بالفكرة والمزاج، ثم حدد السرعة والمقام أو السلم المناسب، وبعدها ابنِ اللحن والكوردات والإيقاع، ثم اكتب الكلمات وسجل نسخة أولية وعدّلها.",
    ),
    ex(
        "music",
        "production",
        "ما العناصر الأساسية في إنتاج أغنية؟",
        "اللحن، الإيقاع، الهارموني، الصوت الرئيسي، الآلات، التسجيل، التوزيع، المكساج والماسترينغ.",
    ),
]


# ============================================================
# 10. DIALOGUE / MULTI-TURN
# ============================================================

DATA["neura_dialogue.jsonl"] = [
    {
        "metadata": {
            "domain": "dialogue",
            "category": "multi_turn",
            "language": "ar",
        },
        "messages": [
            msg("system", "أنت Neura، مساعد ودود ودقيق."),
            msg("user", "أريد تعلم البرمجة."),
            msg("assistant", "ممتاز. هل تريد البدء بـPython أم لديك لغة معينة في بالك؟"),
            msg("user", "لا أعرف أي لغة."),
            msg("assistant", "أنصح غالبًا بـPython كبداية لأنها واضحة وتستخدم في تطبيقات كثيرة مثل الأتمتة والبيانات والذكاء الاصطناعي."),
            msg("user", "كم أحتاج حتى أبدأ مشروعًا؟"),
            msg("assistant", "يمكنك البدء بمشروع صغير خلال أيام إذا ركزت على الأساسيات، ثم تتعلم المفاهيم التي تحتاجها أثناء بناء المشروع."),
        ],
    },
    {
        "metadata": {
            "domain": "dialogue",
            "category": "problem_solving",
            "language": "ar-IQ",
        },
        "messages": [
            msg("system", "أنت Neura، مساعد يتحدث العربية والعراقية عند الحاجة."),
            msg("user", "عندي خطأ بالكود."),
            msg("assistant", "أرسل رسالة الخطأ والكود المرتبط بيها، وأحاول أحدد السبب."),
            msg("user", "الخطأ يقول NameError."),
            msg("assistant", "غالبًا البرنامج يحاول استخدام اسم متغير أو دالة غير معرّفة في ذلك النطاق. أرسل السطر الذي ظهر عنده الخطأ حتى نحدد السبب بدقة."),
        ],
    },
]


# ============================================================
# 11. PROBLEM SOLVING
# ============================================================

DATA["neura_problem_solving.jsonl"] = [
    ex(
        "problem_solving",
        "technical",
        "عندي برنامج بطيء، ماذا أفعل؟",
        "حدد أولًا أين يحدث البطء باستخدام profiling أو قياسات زمنية. لا تحسن الكود بالتخمين. بعد تحديد الجزء البطيء، افحص الخوارزمية، الوصول إلى البيانات، I/O، الذاكرة والتوازي.",
    ),
    ex(
        "problem_solving",
        "planning",
        "كيف تحول فكرة كبيرة إلى مشروع قابل للتنفيذ؟",
        "قسّمها إلى هدف واضح، متطلبات أساسية، MVP صغير، مكونات مستقلة، اختبارات، ثم مراحل تطوير. ابدأ بأصغر نسخة تثبت الفكرة قبل إضافة الميزات المتقدمة.",
    ),
    ex(
        "problem_solving",
        "reasoning",
        "ما أفضل طريقة لحل مشكلة غير واضحة؟",
        "عرّف المشكلة بدقة، اجمع المعطيات، افصل الحقائق عن الافتراضات، ضع عدة فرضيات، اختبرها، ثم اختر الحل بناءً على الأدلة والقيود.",
    ),
]


# ============================================================
# VALIDATION + WRITE
# ============================================================

def fingerprint(item: dict) -> str:
    data = json.dumps(
        item["messages"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def validate_item(item: dict) -> None:
    assert isinstance(item, dict)
    assert "metadata" in item
    assert "messages" in item

    metadata = item["metadata"]

    assert isinstance(metadata.get("domain"), str)
    assert isinstance(metadata.get("category"), str)
    assert isinstance(metadata.get("language"), str)

    messages = item["messages"]

    assert len(messages) >= 2
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "assistant"

    roles = [m["role"] for m in messages]
    assert "user" in roles

    for message in messages:
        assert message["role"] in {
            "system",
            "user",
            "assistant",
        }
        assert isinstance(message["content"], str)
        assert message["content"].strip()


def main():
    all_examples = []
    global_seen = set()

    print("===== NEURA MULTIDOMAIN SFT INJECTION =====")

    for filename, examples in DATA.items():
        path = ROOT / filename

        unique = []

        for item in examples:
            validate_item(item)

            fp = fingerprint(item)

            if fp not in global_seen:
                global_seen.add(fp)
                unique.append(item)
                all_examples.append(item)

        with path.open("w", encoding="utf-8") as f:
            for item in unique:
                f.write(
                    json.dumps(
                        item,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        print(f"{filename:<36} {len(unique):>4} examples")

    merged = ROOT / "neura_multidomain_all.jsonl"

    with merged.open("w", encoding="utf-8") as f:
        for item in all_examples:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print("===== SUMMARY =====")
    print(f"total examples: {len(all_examples)}")
    print(f"unique examples: {len(global_seen)}")
    print(f"duplicates: {len(all_examples) - len(global_seen)}")

    domains = {}
    roles = {}

    for item in all_examples:
        domain = item["metadata"]["domain"]
        domains[domain] = domains.get(domain, 0) + 1

        for message in item["messages"]:
            role = message["role"]
            roles[role] = roles.get(role, 0) + 1

    print()
    print("domains:")
    for name, count in sorted(domains.items()):
        print(f"  {name:<20}: {count}")

    print()
    print("roles:")
    for role, count in sorted(roles.items()):
        print(f"  {role:<12}: {count}")

    print()
    print(f"merged: {merged}")
    print("VALIDATION: PASS")
    print("INJECTION: PASS")


if __name__ == "__main__":
    main()
