"""
Neura Wikipedia Retriever

Lightweight lexical RAG retrieval over Wikipedia.

Pipeline:
    query
      -> query normalization
      -> Wikipedia search
      -> candidate ranking
      -> fetch limited articles
      -> chunk
      -> passage ranking
      -> top-k
"""

import re
import unicodedata
from dataclasses import dataclass

from neura.tools.wikipedia import WikipediaTool


@dataclass
class WikipediaChunk:
    title: str
    text: str
    url: str
    page_id: int | None
    score: float
    language: str


class WikipediaRetriever:

    # Common Arabic function/interrogative words that should not
    # dominate lexical retrieval.
    ARABIC_STOPWORDS = {
        "ما",
        "ماذا",
        "متى",
        "أين",
        "اين",
        "كيف",
        "لماذا",
        "هل",
        "من",
        "مَن",
        "هو",
        "هي",
        "هم",
        "هن",
        "هذا",
        "هذه",
        "ذلك",
        "تلك",
        "الذي",
        "التي",
        "الذين",
        "اللاتي",
        "اللواتي",
        "عن",
        "في",
        "من",
        "إلى",
        "الى",
        "على",
        "مع",
        "هل",
        "قد",
        "لقد",
        "كان",
        "كانت",
        "يكون",
        "تكون",
        "ماهي",
        "ماهو",
    }

    def __init__(
        self,
        wikipedia=None,
        chunk_size=1200,
        overlap=200,
        fetch_candidates=2,
    ):
        self.wikipedia = (
            wikipedia
            if wikipedia is not None
            else WikipediaTool()
        )

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero"
            )

        if overlap < 0:
            raise ValueError(
                "overlap must not be negative"
            )

        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size"
            )

        if fetch_candidates <= 0:
            raise ValueError(
                "fetch_candidates must be greater than zero"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.fetch_candidates = fetch_candidates

    @staticmethod
    def _normalize(text):
        text = str(text or "")

        # Remove HTML.
        text = re.sub(
            r"<[^>]+>",
            " ",
            text,
        )

        # Normalize common Arabic presentation variants.
        text = (
            text
            .replace("أ", "ا")
            .replace("إ", "ا")
            .replace("آ", "ا")
            .replace("ٱ", "ا")
            .replace("ى", "ي")
            .replace("ة", "ة")
        )

        # Remove Arabic diacritics.
        text = re.sub(
            r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]",
            "",
            text,
        )

        # Replace Unicode punctuation and symbols with spaces.
        # This is important for Arabic punctuation such as:
        #   ؟ ، ؛
        # so "العراق؟" becomes "العراق".
        text = "".join(
            " "
            if (
                unicodedata.category(char).startswith("P")
                or unicodedata.category(char).startswith("S")
            )
            else char
            for char in text
        )

        # Normalize whitespace.
        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    @classmethod
    def _terms(cls, text):
        normalized = cls._normalize(text).lower()

        raw_terms = re.findall(
            r"[\w\u0600-\u06ff]+",
            normalized,
            flags=re.UNICODE,
        )

        terms = []

        for term in raw_terms:
            if len(term) < 2:
                continue

            if term in cls.ARABIC_STOPWORDS:
                continue

            if term not in terms:
                terms.append(term)

        return terms

    @staticmethod
    def _term_variants(term):
        """
        Generate conservative Arabic lexical variants.

        This is intentionally small and generic. It must not contain
        bare prefixes such as "عاصم", because substring matching those
        prefixes can inflate scores for unrelated passages.
        """
        variants = {term}

        if term.startswith("عاصم"):
            variants.update(
                {
                    "عاصمة",
                    "عاصمته",
                    "عاصمتها",
                    "عاصمتهم",
                    "عاصمتهما",
                    "عاصمتي",
                }
            )

        return variants

    @classmethod
    def _term_match_count(cls, text, term):
        """
        Count a term and its conservative lexical variants.
        """
        variants = cls._term_variants(term)

        return sum(
            text.count(variant)
            for variant in variants
        )

    def _build_search_queries(self, query):
        """
        Build a small set of normalized Wikipedia search queries.

        The goal is to remove Arabic interrogative scaffolding while
        preserving the actual information-bearing terms. This is a
        retrieval transformation, not an answer map.
        """
        normalized = self._normalize(query).strip()

        if not normalized:
            return []

        terms = self._terms(normalized)

        queries = []

        if terms:
            queries.append(" ".join(terms))

        # Keep the original normalized query as a fallback.
        if normalized not in queries:
            queries.append(normalized)

        # For location questions, a compact "entity + relation"
        # formulation often matches Wikipedia titles/snippets better.
        if len(terms) >= 2:
            for i, term in enumerate(terms):
                if term in {"تقع", "يقع", "موقع", "مكان"}:
                    others = [t for j, t in enumerate(terms) if j != i]
                    if others:
                        queries.insert(0, " ".join(others + [term]))

        # Preserve order while removing duplicates.
        return list(dict.fromkeys(q for q in queries if q.strip()))

    def _score_candidate(
        self,
        query,
        result,
    ):
        query_terms = self._terms(query)

        if not query_terms:
            return 0.0

        title = self._normalize(
            result.get("title", "")
        ).lower()

        snippet = self._normalize(
            result.get("snippet", "")
        ).lower()

        score = 0.0
        matched_terms = 0

        for term in query_terms:
            title_count = self._term_match_count(
                title,
                term,
            )

            snippet_count = self._term_match_count(
                snippet,
                term,
            )

            if title_count:
                matched_terms += 1
            elif snippet_count:
                matched_terms += 1

            # A query concept appearing in the title is much stronger
            # evidence than repeated occurrences in a generic snippet.
            if title_count:
                score += 12.0

            # Limit snippet contribution so repetition cannot dominate.
            if snippet_count:
                score += min(snippet_count, 2) * 1.0

        # Distinct concept coverage.
        score += matched_terms * 4.0

        # Exact title match with one meaningful query concept.
        for term in query_terms:
            if term == title:
                score += 30.0

        # If the whole normalized query is the title, this is extremely
        # strong evidence that the article directly represents the query.
        query_normalized = self._normalize(query).lower()

        if query_normalized and query_normalized == title:
            score += 35.0
        elif query_normalized and query_normalized in title:
            score += 15.0

        # Exact query phrase in the snippet is useful, but deliberately
        # weaker than title evidence.
        if query_normalized and query_normalized in snippet:
            score += 3.0

        return score

    def _score_passage(self, query: str, passage: str) -> float:
        query_terms = self._terms(query)
        normalized = self._normalize(passage)

        if not query_terms:
            return 0.0

        # Resolve each query term to a small set of Arabic variants.
        term_groups = {}
        for term in query_terms:
            variants = set(self._term_variants(term))

            # Definite form is useful for Arabic nouns such as:
            # عاصمة -> العاصمة
            if term.startswith("عاصم"):
                variants.add("العاصمة")

            variants.add(term)
            term_groups[term] = variants

        positions = {}
        matched_variants = {}

        for term, variants in term_groups.items():
            found = []

            for variant in variants:
                if not variant:
                    continue

                # Arabic words can carry short attached clitics:
                # "وعاصمتها", "فالعاصمة", etc.
                # Match the concept itself and record its own position.
                pattern = (
                    r"(?<![\u0600-\u06ff])"
                    r"(?:[وف]\s*)?"
                    r"("
                    + re.escape(variant)
                    + r")"
                    r"(?![\u0600-\u06ff])"
                )

                for match in re.finditer(
                    pattern,
                    normalized,
                    flags=re.UNICODE,
                ):
                    found.append((match.start(1), variant))

            if found:
                found.sort(key=lambda item: item[0])
                positions[term] = [pos for pos, _ in found]
                matched_variants[term] = found[0][1]

        if not positions:
            return 0.0

        score = 0.0

        # 1. Distinct-term coverage.
        score += len(positions) * 5.0

        # 2. Strong bonus when every query concept is represented.
        if len(positions) == len(query_terms):
            score += 10.0

        # 3. Limited repetition bonus.
        for term, term_positions in positions.items():
            score += min(len(term_positions), 3) * 0.5

        # 4. Find ONE closest pair between different query concepts.
        if len(positions) >= 2:
            terms = list(positions.keys())
            best_distance = None

            for i in range(len(terms)):
                for j in range(i + 1, len(terms)):
                    a_positions = positions[terms[i]]
                    b_positions = positions[terms[j]]

                    for a in a_positions:
                        for b in b_positions:
                            distance = abs(a - b)

                            if (
                                best_distance is None
                                or distance < best_distance
                            ):
                                best_distance = distance

            if best_distance is not None:
                if best_distance <= 40:
                    score += 20.0
                elif best_distance <= 80:
                    score += 15.0
                elif best_distance <= 160:
                    score += 10.0
                elif best_distance <= 320:
                    score += 5.0

        # 5. Generic Arabic relational support.
        #
        # Forms such as:
        #   عاصمته
        #   عاصمتها
        #   عاصمتهم
        #
        # carry an explicit possessive relationship. When such a
        # relational form occurs close to another query concept,
        # it is stronger evidence than merely seeing both words
        # somewhere in the same large passage.
        relational_variants = {
            "عاصمته",
            "عاصمتها",
            "عاصمتهم",
            "عاصمتهما",
            "عاصمتي",
        }

        for term, variants in term_groups.items():
            relational_positions = []

            for variant in variants:
                if variant not in relational_variants:
                    continue

                pattern = (
                    r"(?<![\u0600-\u06ff])"
                    r"(?:[وف]\s*)?"
                    r"("
                    + re.escape(variant)
                    + r")"
                    r"(?![\u0600-\u06ff])"
                )

                for match in re.finditer(
                    pattern,
                    normalized,
                    flags=re.UNICODE,
                ):
                    relational_positions.append(match.start(1))

            if not relational_positions:
                continue

            # Compare the relational expression against all other
            # query concepts.
            for other_term, other_positions in positions.items():
                if other_term == term:
                    continue

                closest = min(
                    abs(a - b)
                    for a in relational_positions
                    for b in other_positions
                )

                if closest <= 40:
                    score += 15.0
                elif closest <= 80:
                    score += 8.0
                elif closest <= 160:
                    score += 3.0

        # Strong semantic signal for possessive capital forms.
        # Example:
        #   "السعودية ... وعاصمتها مدينة الرياض"
        # This is stronger evidence for "عاصمة السعودية" than
        # merely finding "عاصمة" and "السعودية" independently.
        if len(query_terms) >= 2:
            capital_forms = {
                "عاصمته",
                "عاصمتها",
                "عاصمتهم",
                "عاصمتهما",
                "عاصمتي",
            }

            for capital_form in capital_forms:
                pattern = (
                    r"(?<![\u0600-\u06ff])"
                    r"(?:[وف]\s*)?"
                    r"("
                    + re.escape(capital_form)
                    + r")"
                    r"(?![\u0600-\u06ff])"
                )

                capital_positions = [
                    match.start(1)
                    for match in re.finditer(
                        pattern,
                        normalized,
                        flags=re.UNICODE,
                    )
                ]

                if not capital_positions:
                    continue

                other_positions = []

                for term in query_terms:
                    if term.startswith("عاصم"):
                        continue

                    other_positions.extend(
                        positions.get(term, [])
                    )

                if not other_positions:
                    continue

                closest = min(
                    abs(a - b)
                    for a in capital_positions
                    for b in other_positions
                )

                if closest <= 80:
                    score += 25.0
                elif closest <= 160:
                    score += 20.0
                elif closest <= 260:
                    score += 15.0
                elif closest <= 400:
                    score += 8.0

        return score

    def _chunk(self, text):
        text = self._normalize(text)

        if not text:
            return []

        chunks = []

        start = 0
        length = len(text)

        while start < length:
            end = min(
                start + self.chunk_size,
                length,
            )

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= length:
                break

            start = end - self.overlap

        return chunks

    def _rank_search_results(
        self,
        query,
        results,
    ):
        ranked = []

        for index, result in enumerate(results):
            score = self._score_candidate(
                query,
                result,
            )

            ranked.append(
                (
                    score,
                    index,
                    result,
                )
            )

        # Stable ordering: score first, original Wikipedia
        # result order second.
        ranked.sort(
            key=lambda item: (
                item[0],
                -item[1],
            ),
            reverse=True,
        )

        return [
            result
            for _, _, result in ranked
        ]

    def retrieve_article(
        self,
        query,
        title,
        language="ar",
        top_k=5,
    ):
        article = self.wikipedia.get(
            title,
            language=language,
        )

        if article is None:
            return []

        # WikipediaTool returns article content as "extract".
        text = article.get(
            "extract",
            "",
        )

        chunks = self._chunk(text)

        results = []

        for chunk in chunks:
            score = self._score_passage(
                query,
                chunk,
            )

            results.append(
                WikipediaChunk(
                    title=article.get(
                        "title",
                        title,
                    ),
                    text=chunk,
                    url=article.get(
                        "url",
                        "",
                    ),
                    page_id=article.get(
                        "page_id",
                    ),
                    score=score,
                    language=article.get(
                        "language",
                        language,
                    ),
                )
            )

        results.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return results[:top_k]

    def retrieve(
        self,
        query,
        language="ar",
        max_results=10,
        top_k=5,
    ):
        """
        Retrieve relevant Wikipedia passages.

        Search candidates are merged across generated query variants.
        Candidate selection uses both lexical evidence and Wikipedia
        search rank, while final ranking is performed on article passages.
        """
        search_queries = self._build_search_queries(query)

        combined = {}

        # Keep several candidates from every query variant.
        per_query_limit = min(max_results, 5)

        for search_query in search_queries:
            try:
                results = self.wikipedia.search(
                    search_query,
                    language=language,
                    max_results=max_results,
                )
            except Exception:
                continue

            for result_index, result in enumerate(
                results[:per_query_limit]
            ):
                result = dict(result)

                page_id = result.get("page_id")
                title = result.get("title", "")

                key = (
                    f"page:{page_id}"
                    if page_id is not None
                    else f"title:{title}"
                )

                search_score = self._score_candidate(
                    search_query,
                    result,
                )

                old = combined.get(key)

                if old is None:
                    result["_best_search_score"] = search_score
                    result["_best_search_rank"] = result_index
                    result["_best_search_query"] = search_query
                    combined[key] = result
                    continue

                old_score = old.get(
                    "_best_search_score",
                    float("-inf"),
                )
                old_rank = old.get(
                    "_best_search_rank",
                    max_results,
                )

                if (
                    search_score > old_score
                    or (
                        search_score == old_score
                        and result_index < old_rank
                    )
                ):
                    result["_best_search_score"] = search_score
                    result["_best_search_rank"] = result_index
                    result["_best_search_query"] = search_query
                    combined[key] = result

        if not combined:
            return []

        ranked_candidates = []

        for result in combined.values():
            search_score = float(
                result.get("_best_search_score", 0.0)
            )

            search_rank = int(
                result.get("_best_search_rank", max_results)
            )

            rank_bonus = max(
                0.0,
                float(per_query_limit - search_rank),
            ) * 3.0

            candidate_score = search_score + rank_bonus

            ranked_candidates.append(
                (
                    candidate_score,
                    search_score,
                    search_rank,
                    result,
                )
            )

        ranked_candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
                -item[2],
            ),
            reverse=True,
        )

        # Keep enough candidates for passage-level ranking.
        fetch_limit = max(
            self.fetch_candidates,
            len(search_queries) * 2,
        )

        selected = ranked_candidates[:fetch_limit]

        all_chunks = []

        for (
            candidate_score,
            search_score,
            search_rank,
            result,
        ) in selected:
            title = result.get("title", "")

            if not title:
                continue

            try:
                article = self.wikipedia.get(
                    title,
                    language=language,
                )
            except Exception:
                continue

            article_text = article.get("extract", "")

            if not article_text:
                continue

            raw_chunks = self._chunk(article_text)

            article_title = article.get(
                "title",
                title,
            )

            article_url = article.get(
                "url",
                "",
            )

            article_page_id = article.get(
                "page_id",
                result.get("page_id"),
            )

            # Candidate evidence is only a small prior.
            candidate_bonus = min(
                max(candidate_score, 0.0),
                100.0,
            ) * 0.05

            for chunk_text in raw_chunks:
                passage_score = self._score_passage(
                    query,
                    chunk_text,
                )

                all_chunks.append(
                    WikipediaChunk(
                        title=article_title,
                        text=chunk_text,
                        url=article_url,
                        page_id=article_page_id,
                        score=passage_score + candidate_bonus,
                        language=language,
                    )
                )

        all_chunks.sort(
            key=lambda chunk: chunk.score,
            reverse=True,
        )

        return all_chunks[:top_k]
