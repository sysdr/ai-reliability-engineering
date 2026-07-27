"""
Day 1 - Architecture: The Five-Component Pipeline

Every later phase of this course (eval harness, deployment safety, audit
trail, cost governance) attaches to the seams between these five stages.
Today, each stage is deterministic and simple on purpose - the goal is to
prove the seams work before any single stage gets smarter.
"""

from dataclasses import dataclass, field


@dataclass
class Query:
    text: str
    session_id: str


@dataclass
class Intent:
    query: Query
    intent_label: str


@dataclass
class RetrievedContext:
    intent: Intent
    passages: list[str] = field(default_factory=list)


@dataclass
class DraftAnswer:
    context: RetrievedContext
    text: str


@dataclass
class CriticVerdict:
    draft: DraftAnswer
    approved: bool
    reason: str


@dataclass
class FinalResponse:
    text: str
    approved: bool


class QueryUnderstandingStage:
    KNOWN_INTENTS = {
        "refund": "refund_policy",
        "cancel": "cancellation",
        "price": "pricing",
    }

    def run(self, query: Query) -> Intent:
        lowered = query.text.lower()
        label = next(
            (v for k, v in self.KNOWN_INTENTS.items() if k in lowered),
            "general_inquiry",
        )
        return Intent(query=query, intent_label=label)


class RetrievalStage:
    MOCK_KNOWLEDGE_BASE = {
        "refund_policy": [
            "Annual plans can be refunded within 30 days of purchase.",
            "Monthly plans are non-refundable after the billing date.",
        ],
        "cancellation": ["Cancel anytime from account settings; no fees."],
        "pricing": ["Pricing starts at $39/month or $279.30/year."],
        "general_inquiry": ["Please check our help center for details."],
    }

    def run(self, intent: Intent) -> RetrievedContext:
        passages = self.MOCK_KNOWLEDGE_BASE.get(intent.intent_label, [])
        return RetrievedContext(intent=intent, passages=passages)


class SynthesisStage:
    def run(self, context: RetrievedContext) -> DraftAnswer:
        if not context.passages:
            text = "I don't have enough information to answer that yet."
        else:
            text = context.passages[0]
        return DraftAnswer(context=context, text=text)


class CriticStage:
    def run(self, draft: DraftAnswer) -> CriticVerdict:
        if not draft.text.strip():
            return CriticVerdict(draft=draft, approved=False, reason="empty answer")
        if draft.context.intent.intent_label == "general_inquiry":
            return CriticVerdict(draft=draft, approved=False, reason="low confidence intent")
        return CriticVerdict(draft=draft, approved=True, reason="passes basic checks")


class FormattingStage:
    def run(self, verdict: CriticVerdict) -> FinalResponse:
        text = verdict.draft.text if verdict.approved else (
            "I'm not confident enough to answer that - routing to a human."
        )
        return FinalResponse(text=text, approved=verdict.approved)


class Pipeline:
    def __init__(self):
        self.query_understanding = QueryUnderstandingStage()
        self.retrieval = RetrievalStage()
        self.synthesis = SynthesisStage()
        self.critic = CriticStage()
        self.formatter = FormattingStage()

    def run(self, query: Query) -> FinalResponse:
        intent = self.query_understanding.run(query)
        print(f"Stage 1 (Query Understanding) -> intent: {intent.intent_label}")

        context = self.retrieval.run(intent)
        print(f"Stage 2 (Retrieval) -> {len(context.passages)} passages found")

        draft = self.synthesis.run(context)
        print("Stage 3 (Synthesis) -> draft answer generated")

        verdict = self.critic.run(draft)
        print(f"Stage 4 (Critic) -> approved: {verdict.approved}")

        response = self.formatter.run(verdict)
        print("Stage 5 (Formatting) -> response ready")

        return response


if __name__ == "__main__":
    query = Query(
        text="What's the refund policy for annual plans?",
        session_id="demo-session-001",
    )
    print(f'Query received: "{query.text}"')

    pipeline = Pipeline()
    result = pipeline.run(query)

    print(f'Final: "{result.text}"')
