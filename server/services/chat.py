"""
Neura Server Chat Service
=========================

Chat service with persistent conversation storage.

The service keeps:
- user profile / interests
- persistent conversations
- persistent messages

The SQLite ConversationStore is the authoritative storage
for conversation history.
"""

from __future__ import annotations

import json
import os
import re
from html import unescape

from neura.memory.conversation_store import ConversationStore
from neura.memory.learning_memory import LearningMemory
from neura.tokenizer import TOKENIZER
from neura.inference.sft_runtime import SFTRuntime
from neura.agents.orchestrator import NeuraOrchestrator
from neura.tools.web_fetch import WebFetch
from server.services.smart_router import SMART_ROUTER


class ChatServiceError(Exception):
    """Base exception for chat service errors."""


class ChatService:
    """
    Neura HTTP chat service.

    Conversation persistence:

        server
            ↓
        ChatService
            ↓
        ConversationStore
            ↓
        data/neura.db
    """

    def __init__(
        self,
        tokenizer=None,
        model=None,
        generator=None,
        conversation_store=None,
        user_id=None,
    ):
        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else TOKENIZER
        )

        self.model = model
        self.generator = generator

        # Trusted server-side identity.
        #
        # IMPORTANT:
        # This value must come from the server/authentication layer,
        # never from the client payload.
        self.user_id = (
            str(user_id).strip()
            if user_id is not None
            else None
        )

        # Real SFT inference runtime.
        #
        # The current SFT checkpoint is intentionally loaded through
        # SFTRuntime because NeuraTorchModel does not yet implement
        # the KV-cache interface required by InferenceEngine.
        self.sft_runtime = None

        try:
            self.sft_runtime = SFTRuntime(
                "checkpoints/neura_chat_sft_best.pt",
                device="cpu",
            )

            print(
                "[+] SFT Runtime Active."
            )
            print(
                "[+] SFT Vocabulary: "
                f"{self.sft_runtime.vocab_size}"
            )

        except Exception as exc:
            print(
                "[Neura SFT Runtime Warning] "
                f"{exc!r}"
            )

        # --------------------------------------------------
        # Neura Agent Orchestrator
        # --------------------------------------------------
        # Tool-backed requests can use GitHub and official-site
        # integrations without replacing the existing SFT path.
        self.orchestrator = NeuraOrchestrator(
            model_callback=self._generate_orchestrated_prompt,
        )

        # --------------------------------------------------
        # Multilingual Smart Router
        # --------------------------------------------------
        self.smart_router = SMART_ROUTER
        self.web_fetcher = WebFetch(timeout=10)

        # Persistent conversation storage.
        self.conversation_store = (
            conversation_store
            if conversation_store is not None
            else ConversationStore()
        )
        # Persistent learning memory.
        # Uses the same SQLite database as ConversationStore.
        self.learning_memory = LearningMemory("data/neura.db")


        # Compatibility cache.
        #
        # This is NOT the source of truth anymore.
        # SQLite is authoritative.
        self.history = []

        # Persistent user profile / interests.
        self.user_profile_path = (
            "data/user_profile.json"
        )

        self.user_interests = (
            self._load_user_profile()
        )

        print(
            "[+] Persistent Conversation Store Active."
        )
        print(
            "[+] Database: data/neura.db"
        )

    def _generate_orchestrated_prompt(self, prompt):
        """
        Generate the final answer for tool-backed requests.

        The Orchestrator supplies a prompt containing the user's
        question plus retrieved web/tool context.

        The existing SFT runtime remains the actual generator.
        """
        if self.sft_runtime is None:
            return ""

        try:
            result = self.sft_runtime.generate(
                prompt,
                max_new_tokens=64,
                temperature=0.0,
                repetition_penalty=1.18,
                no_repeat_ngram_size=2,
                repeat_token_limit=3,
            )

            response = result.text.strip()

            for marker in (
                "<EOS>",
                "<|end|>",
                "<|assistant|>",
                "<|user|>",
                "<|system|>",
                "<BOS>",
            ):
                response = response.replace(
                    marker,
                    "",
                ).strip()

            return response

        except Exception as exc:
            print(
                "[Neura Orchestrator Generation Error] "
                f"{exc!r}"
            )
            return ""


    # ==========================================================
    # USER PROFILE
    # ==========================================================

    def _load_user_profile(self) -> dict:
        if os.path.exists(
            self.user_profile_path
        ):
            try:
                with open(
                    self.user_profile_path,
                    "r",
                    encoding="utf-8",
                ) as file:
                    profile = json.load(file)

                if not isinstance(profile, dict):
                    return {}

                profile.setdefault(
                    "preferred_topics",
                    [],
                )
                profile.setdefault(
                    "user_name",
                    "",
                )

                return profile

            except (
                OSError,
                json.JSONDecodeError,
            ):
                return {}

        return {
            "preferred_topics": [],
            "user_name": "",
        }

    def _save_user_profile(self) -> None:
        os.makedirs(
            "data",
            exist_ok=True,
        )

        with open(
            self.user_profile_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.user_interests,
                file,
                ensure_ascii=False,
                indent=4,
            )

    def _extract_learning_signals(
        self,
        text: str,
    ) -> None:
        if "اسمي " in text:
            parts = text.split("اسمي ")

            if len(parts) > 1:
                name = (
                    parts[1]
                    .split()[0]
                    .strip(".,!؟")
                )

                if name:
                    self.user_interests[
                        "user_name"
                    ] = name

        keywords = [
            "برمجة",
            "ذكاء اصطناعي",
            "تطوير",
            "موقع",
            "بايثون",
            "تصميم",
            "تعديل",
        ]

        topics = self.user_interests.setdefault(
            "preferred_topics",
            [],
        )

        for keyword in keywords:
            if (
                keyword in text
                and keyword not in topics
            ):
                topics.append(keyword)

        self._save_user_profile()

    # ==========================================================
    # CONVERSATION HELPERS
    # ==========================================================

    def _create_conversation(
        self,
        user_id: str | None = None,
    ) -> str:
        resolved_user_id = (
            user_id
            if user_id is not None
            else self.user_id
        )

        if not resolved_user_id:
            raise ChatServiceError(
                "Authenticated user identity is required."
            )

        return (
            self.conversation_store
            .create_conversation(
                user_id=resolved_user_id,
            )
        )

    def _load_history(
        self,
        conversation_id: str,
    ) -> list[dict]:
        messages = (
            self.conversation_store
            .get_messages(
                conversation_id
            )
        )

        return [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in messages
        ]

    def _conversation_exists(
        self,
        conversation_id: str,
    ) -> bool:
        return (
            self.conversation_store
            .conversation_exists(
                conversation_id
            )
        )

    def _conversation_belongs_to_user(
        self,
        conversation_id: str,
    ) -> bool:
        if not self.user_id:
            return False

        return self.conversation_store.conversation_belongs_to_user(
            conversation_id,
            self.user_id,
        )

    # ==========================================================
    # CHAT
    # ==========================================================

    def _clean_web_text(self, text: str, limit: int = 6000) -> str:
        """Extract readable text from HTML/text."""
        text = str(text or "")

        text = re.sub(
            r"(?is)<(script|style|noscript).*?>.*?</\1>",
            " ",
            text,
        )
        text = re.sub(r"(?s)<!--.*?-->", " ", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)

        text = unescape(text)
        text = re.sub(r"\s+", " ", text).strip()

        return text[:limit]

    def _fetch_url_response(
        self,
        url: str,
        language: str = "en",
    ) -> str:
        """Fetch a public URL and return bounded readable text."""
        try:
            result = self.web_fetcher.fetch(url)

            status = int(result.get("status", 0))
            body = self._clean_web_text(
                result.get("body", "")
            )

            if status < 200 or status >= 400:
                if language == "ar":
                    return (
                        f"تعذر فتح الرابط. "
                        f"رمز الاستجابة HTTP: {status}."
                    )
                return (
                    f"Unable to open the URL. "
                    f"HTTP status: {status}."
                )

            if not body:
                if language == "ar":
                    return (
                        "تم فتح الرابط، لكن لم أجد "
                        "نصاً قابلاً للقراءة فيه."
                    )
                return (
                    "The URL was opened, but no readable "
                    "text was found."
                )

            if language == "ar":
                return (
                    "تم فتح الرابط. هذا النص المستخرج منه:\n\n"
                    + body
                )

            return (
                "The URL was opened. Extracted text:\n\n"
                + body
            )

        except Exception as exc:
            print(
                "[SmartRouter URL Error] "
                f"{exc!r}"
            )

            if language == "ar":
                return "تعذر الوصول إلى الرابط حالياً."

            return "Unable to access the URL right now."

    def handle_chat(
        self,
        payload: dict,
    ) -> dict:
        if not isinstance(payload, dict):
            raise ChatServiceError(
                "payload must be a dictionary"
            )

        user_message = ""

        if "message" in payload:
            user_message = str(
                payload.get(
                    "message",
                    "",
                )
            ).strip()

        elif "messages" in payload:
            messages_list = payload.get(
                "messages",
                [],
            )

            if messages_list:
                last_message = (
                    messages_list[-1]
                )

                if isinstance(
                    last_message,
                    dict,
                ):
                    user_message = str(
                        last_message.get(
                            "content",
                            "",
                        )
                    ).strip()

        if not user_message:
            return {
                "status": "error",
                "response": (
                    "لم يتم التعرف على نص الرسالة."
                ),
            }

        # ------------------------------------------------------
        # Conversation identity
        # ------------------------------------------------------

        conversation_id = payload.get(
            "conversation_id"
        )

        if conversation_id is not None:
            conversation_id = str(
                conversation_id
            ).strip()

        if not self.user_id:
            return {
                "status": "error",
                "response": (
                    "هوية المستخدم غير متوفرة على الخادم."
                ),
                "conversation_id": conversation_id,
            }

        if not conversation_id:
            conversation_id = (
                self._create_conversation()
            )

        elif not self._conversation_exists(
            conversation_id
        ):
            return {
                "status": "error",
                "response": (
                    "المحادثة المطلوبة غير موجودة."
                ),
                "conversation_id": (
                    conversation_id
                ),
            }

        elif not self._conversation_belongs_to_user(
            conversation_id
        ):
            return {
                "status": "error",
                "response": (
                    "غير مسموح بالوصول إلى هذه المحادثة."
                ),
                "conversation_id": (
                    conversation_id
                ),
            }

        # ------------------------------------------------------
        # Load persistent history
        # ------------------------------------------------------

        self.history = self._load_history(
            conversation_id
        )

        # ------------------------------------------------------
        # Learning/profile
        # ------------------------------------------------------

        self._extract_learning_signals(
            user_message.lower()
        )

        # ------------------------------------------------------
        # Persist user message FIRST
        # ------------------------------------------------------

        # Keep the history from BEFORE the current user
        # message. This is the actual previous context.
        previous_history = list(self.history)

        self.conversation_store.add_message(
            conversation_id=conversation_id,
            role="user",
            content=user_message,
        )

        # Update in-memory compatibility history.
        self.history.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        # ------------------------------------------------------
        # Persistent Learning Memory
        # ------------------------------------------------------

        learned_memories = []

        try:
            learned_memories = self.learning_memory.learn_from_message(
                user_id=self.user_id,
                text=user_message,
                conversation_id=conversation_id,
            )

            if learned_memories:
                print(
                    "[LearningMemory] "
                    f"learned={len(learned_memories)} "
                    f"conversation={conversation_id}"
                )

        except Exception as exc:
            print(
                "[LearningMemory Error] "
                f"{exc!r}"
            )

        # ------------------------------------------------------
        # Response generation
        # ------------------------------------------------------

        user_name = self.user_interests.get(
            "user_name",
            "",
        )

        fav_topics = self.user_interests.get(
            "preferred_topics",
            [],
        )

        msg_lower = user_message.lower().strip()

        # --------------------------------------------------
        # Multilingual Smart Router
        # --------------------------------------------------
        # "model" deliberately falls through to the existing
        # deterministic/profile/Orchestrator/SFT pipeline.
        bot_response = ""

        smart_route = self.smart_router.route(user_message)

        print(
            "[SmartRouter] "
            f"action={smart_route.action} "
            f"language={smart_route.language} "
            f"confidence={smart_route.confidence:.2f}"
        )

        if smart_route.action in (
            "greeting",
            "smalltalk",
            "identity",
            "knowledge",
            "programming",
            "unsafe_url",
        ):
            bot_response = smart_route.response.strip()

        elif smart_route.action == "url":
            bot_response = self._fetch_url_response(
                smart_route.url,
                smart_route.language,
            )

        elif smart_route.action == "github":
            try:
                orchestrated = self.orchestrator.run(
                    user_message,
                    max_results=3,
                )

                bot_response = (
                    orchestrated.get("response", "")
                    .strip()
                )

                if not bot_response:
                    bot_response = (
                        "تم تنفيذ طلب GitHub، "
                        "لكن لم يتم إنتاج إجابة."
                    )

                print(
                    "[SmartRouter -> Orchestrator] "
                    "action=github "
                    f"documents="
                    f"{len(orchestrated.get('documents', []))}"
                )

            except Exception as exc:
                print(
                    "[SmartRouter GitHub Error] "
                    f"{exc!r}"
                )

                bot_response = (
                    "حدث خطأ أثناء تنفيذ أداة GitHub في Neura."
                )


        # ردود سريعة ومباشرة للحالات العامة والتحيات.
        # إذا تطابقت الرسالة، نتجاوز مسار التوليد الكامل.
        quick_greetings = {
            "كيف حالك": "بخير، وأنت كيف حالك؟",
            "مرحباً": "أهلاً بك! كيف يمكنني مساعدتك اليوم؟",
            "السلام عليكم": "وعليكم السلام ورحمة الله وبركاته، تفضل.",
        }

        quick_greeting_response = quick_greetings.get(
            msg_lower
        )

        greeting_name = (
            f"يا {user_name}"
            if user_name
            else "عزيزي المطور"
        )

        if bot_response:
            pass
        elif quick_greeting_response is not None:
            bot_response = quick_greeting_response

        elif (
            "من انت" in msg_lower
            or "ما اسمك" in msg_lower
        ):
            bot_response = (
                "أنا نيورا (Neura)، "
                "مساعدك الشخصي للذكاء الاصطناعي. "
                "صُممت خصيصاً لأكون شريكك التقني "
                "في مشاريعك."
            )

            if user_name:
                bot_response += (
                    f" وأنا فخور بالعمل معك "
                    f"{greeting_name}."
                )

        elif (
            "ما اسمي" in msg_lower
            or "هل تتذكرني" in msg_lower
        ):
            if user_name:
                bot_response = (
                    f"بالتأكيد أتذكرك! أنت "
                    f"{user_name}. "
                    "وسجل محادثاتنا محفوظ "
                    "في قاعدة بيانات المحادثات."
                )
            else:
                bot_response = (
                    "أنت لم تخبرني باسمك بعد. "
                    "أخبرني 'اسمي ...' وسأحفظ "
                    "هذه المعلومة في ملف ملفك الشخصي."
                )

        elif (
            "ماذا تعلمت عني" in msg_lower
            or "اهتماماتي" in msg_lower
        ):
            if fav_topics:
                bot_response = (
                    "بناءً على المعلومات التي "
                    "تم حفظها عن اهتماماتك، "
                    "أنت مهتم بـ: "
                    f"({'، '.join(fav_topics)})."
                )
            else:
                bot_response = (
                    "لا توجد اهتمامات محفوظة "
                    "بشكل كافٍ حتى الآن."
                )

        elif (
            "تصفير" in msg_lower
            or "امسح الذاكرة" in msg_lower
        ):
            # Clear persistent messages for this
            # conversation only.
            self.conversation_store.clear_messages(
                conversation_id
            )

            self.history = []

            self.user_interests = {
                "preferred_topics": [],
                "user_name": "",
            }

            self._save_user_profile()

            bot_response = (
                "تم تصفير سجل هذه المحادثة "
                "والاهتمامات المحفوظة. "
                "أنا مستعد لبدء صفحة جديدة معك."
            )

        else:
            # --------------------------------------------------
            # REAL MODEL INFERENCE
            # --------------------------------------------------
            #
            # ConversationStore remains authoritative.
            #
            # previous_history contains the conversation BEFORE
            # the current user message, which was already persisted
            # above.
            #
            # The current SFT model is a tiny proof-of-runtime model.
            # Therefore this path verifies real inference integration;
            # response quality is expected to remain limited until
            # the model is trained on a larger vocabulary/dataset.

            # --------------------------------------------------
            # TOOL-BACKED ORCHESTRATOR REQUEST
            # --------------------------------------------------
            #
            # GitHub and official-site requests are handled by
            # the Orchestrator. Normal chat keeps the existing
            # SFT inference path below.
            #
            # Web-search routing is intentionally disabled here
            # until the WebSearch provider is fixed.
            tool_decision = self.orchestrator.tool_agent.decide(
                user_message
            )

            if tool_decision["action"] in (
                "github",
                "official_site",
                "wikipedia",
                "knowledge_source",
            ):
                try:
                    orchestrated = self.orchestrator.run(
                        user_message,
                        max_results=3,
                    )

                    bot_response = (
                        orchestrated.get(
                            "response",
                            "",
                        ).strip()
                    )

                    if not bot_response:
                        bot_response = (
                            "تم تنفيذ البحث، "
                            "لكن النموذج لم يُنتج إجابة."
                        )

                    print(
                        "[Orchestrator] "
                        f"action={tool_decision['action']} "
                        f"documents="
                        f"{len(orchestrated.get('documents', []))}"
                    )

                except Exception as exc:
                    print(
                        "[Neura Orchestrator Error] "
                        f"{exc!r}"
                    )

                    bot_response = (
                        "حدث خطأ أثناء تنفيذ أدوات Neura."
                    )

            elif self.sft_runtime is not None:
                try:
                    # Build the conversational context.
                    #
                    # Keep the context bounded because the current
                    # SFT checkpoint has context_length=1024.
                    context_messages = previous_history[-12:]

                    prompt_parts = [
                        "<BOS>",
                        "<|system|>",
                        "You are Neura.",
                        "<|end|>",
                    ]

                    for message in context_messages:
                        role = message.get("role")
                        content = str(
                            message.get("content", "")
                        ).strip()

                        if not content:
                            continue

                        if role == "user":
                            prompt_parts.extend([
                                "<|user|>",
                                content,
                                "<|end|>",
                            ])

                        elif role == "assistant":
                            prompt_parts.extend([
                                "<|assistant|>",
                                content,
                                "<|end|>",
                            ])

                    prompt_parts.extend([
                        "<|user|>",
                        user_message,
                        "<|end|>",
                        "<|assistant|>",
                    ])

                    prompt = " ".join(prompt_parts)

                    result = self.sft_runtime.generate(
                        prompt,
                        max_new_tokens=64,
                        temperature=0.0,
                        repetition_penalty=1.18,
                        no_repeat_ngram_size=2,
                        repeat_token_limit=3,
                    )

                    bot_response = result.text.strip()

                    # Remove accidental special markers from the
                    # visible assistant response.
                    for marker in (
                        "<EOS>",
                        "<|end|>",
                        "<|assistant|>",
                        "<|user|>",
                        "<|system|>",
                        "<BOS>",
                    ):
                        bot_response = bot_response.replace(
                            marker,
                            "",
                        ).strip()

                    if not bot_response:
                        bot_response = (
                            "تم تشغيل نموذج Neura، "
                            "لكن النموذج لم يُنتج نصاً."
                        )

                    print(
                        "[SFT Inference] "
                        f"finish={result.finish_reason} "
                        f"generated={len(result.generated_tokens)}"
                    )

                except Exception as exc:
                    print(
                        "[Neura SFT Inference Error] "
                        f"{exc!r}"
                    )

                    bot_response = (
                        "حدث خطأ أثناء تشغيل نموذج Neura."
                    )

            else:
                # Safe fallback if the SFT runtime could not load.
                bot_response = (
                    f"مرحباً بك {greeting_name}. "
                    f"استلمت رسالتك "
                    f"'{user_message}'، "
                    "لكن نموذج Neura غير متاح حالياً."
                )

        # ------------------------------------------------------
        # Persist assistant response
        # ------------------------------------------------------

        self.conversation_store.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=bot_response,
        )

        self.history.append(
            {
                "role": "assistant",
                "content": bot_response,
            }
        )

        print(
            "[Memory Status] "
            f"Conversation: {conversation_id} "
            f"| Messages: "
            f"{self.conversation_store.message_count(conversation_id)}"
        )

        return {
            "status": "ok",
            "response": bot_response,
            "conversation_id": conversation_id,
        }


__all__ = [
    "ChatService",
    "ChatServiceError",
]
