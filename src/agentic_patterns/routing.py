"""
Routing Pattern Implementation.

Based on the Agentic Design Patterns book Chapter 2:
Use an LLM to classify user intent and route queries to specialized handlers.

Example use case: Customer service routing
- Order status queries -> order_status handler
- Product information queries -> product_info handler
- Technical support queries -> technical_support handler
- Unclear queries -> clarification handler
"""

from enum import Enum

from pydantic import BaseModel
from pydantic import Field
from pydantic_ai import Agent

from agentic_patterns._models import get_model


# --8<-- [start:models]
class Intent(str, Enum):
    """Possible intents for routing."""

    ORDER_STATUS = "order_status"
    PRODUCT_INFO = "product_info"
    TECHNICAL_SUPPORT = "technical_support"
    CLARIFICATION = "clarification"


class RouteDecision(BaseModel):
    """Router's decision on where to send the query."""

    intent: Intent = Field(description="The classified intent of the query")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the classification (0-1)",
    )
    reasoning: str = Field(
        description="Brief explanation of why this intent was chosen"
    )


class OrderStatusResponse(BaseModel):
    """Response for order status queries."""

    order_id: str | None = Field(
        default=None, description="Extracted order ID if found"
    )
    status_message: str = Field(description="Status information or guidance")
    next_steps: list[str] = Field(
        default_factory=list, description="Suggested next steps"
    )


class ProductInfoResponse(BaseModel):
    """Response for product information queries."""

    product_name: str | None = Field(
        default=None, description="Product being asked about"
    )
    information: str = Field(description="Product information or guidance")
    related_products: list[str] = Field(
        default_factory=list, description="Related products to suggest"
    )


class TechnicalSupportResponse(BaseModel):
    """Response for technical support queries."""

    issue_category: str = Field(description="Category of the technical issue")
    troubleshooting_steps: list[str] = Field(
        description="Steps to resolve the issue"
    )
    escalation_needed: bool = Field(
        default=False, description="Whether human escalation is needed"
    )


class ClarificationResponse(BaseModel):
    """Response when the query needs clarification."""

    original_query: str = Field(description="The original query received")
    clarifying_questions: list[str] = Field(
        description="Questions to ask for clarification"
    )
    possible_intents: list[str] = Field(
        description="What the user might be asking about"
    )


# Union type for all possible responses
RouteResponse = (
    OrderStatusResponse
    | ProductInfoResponse
    | TechnicalSupportResponse
    | ClarificationResponse
)
# --8<-- [end:models]


# Initialize the model
model = get_model()


# --8<-- [start:agents]
# Router agent - classifies intent
router_agent = Agent(
    model,
    system_prompt=(
        "You are an intent classifier for a customer service system. "
        "Analyze the user's query and determine their intent.\n\n"
        "Classify into one of these categories:\n"
        "- order_status: Questions about order tracking, delivery, shipping\n"
        "- product_info: Questions about products, features, pricing\n"
        "- technical_support: Technical issues, troubleshooting, bugs\n"
        "- clarification: Query is unclear or doesn't fit other categories\n\n"
        "Provide a confidence score (0-1) based on how certain you are. "
        "If confidence is below 0.7, consider using 'clarification' instead."
    ),
    output_type=RouteDecision,
)

# Specialized handlers
order_status_agent = Agent(
    model,
    system_prompt=(
        "You are an order status specialist. "
        "Help customers with their order-related queries. "
        "Extract order IDs if mentioned (format: ORD-XXXX or similar). "
        "Provide clear status information and helpful next steps. "
        "If no order ID is provided, ask the customer to provide it."
    ),
    output_type=OrderStatusResponse,
)

product_info_agent = Agent(
    model,
    system_prompt=(
        "You are a product information specialist. "
        "Help customers learn about products, features, pricing. "
        "Identify the specific product being asked about. "
        "Suggest related products when appropriate. "
        "Be helpful and informative."
    ),
    output_type=ProductInfoResponse,
)

technical_support_agent = Agent(
    model,
    system_prompt=(
        "You are a technical support specialist. "
        "Help customers troubleshoot technical issues. "
        "Categorize the issue type and provide step-by-step troubleshooting. "
        "Determine if the issue needs human escalation (complex hardware, "
        "account security, billing disputes should be escalated). "
        "Keep troubleshooting steps clear and actionable."
    ),
    output_type=TechnicalSupportResponse,
)

clarification_agent = Agent(
    model,
    system_prompt=(
        "You are a customer service assistant. "
        "The user's query was unclear or ambiguous. "
        "Ask clarifying questions to understand what they need. "
        "Suggest what they might be asking about based on common queries. "
        "Be polite and helpful in seeking clarification."
    ),
    output_type=ClarificationResponse,
)

# Map intents to handlers
INTENT_HANDLERS = {
    Intent.ORDER_STATUS: order_status_agent,
    Intent.PRODUCT_INFO: product_info_agent,
    Intent.TECHNICAL_SUPPORT: technical_support_agent,
    Intent.CLARIFICATION: clarification_agent,
}
# --8<-- [end:agents]


# --8<-- [start:routing]
async def route_query(user_query: str) -> tuple[RouteDecision, RouteResponse]:
    """
    Route a user query to the appropriate handler.

    Args:
        user_query: The user's question or request.

    Returns:
        Tuple of (routing decision, handler response).
    """
    # Step 1: Classify intent
    print(f"Routing query: {user_query[:50]}...")
    route_result = await router_agent.run(
        f"Classify the intent of this customer query:\n\n{user_query}"
    )
    decision = route_result.output
    conf = decision.confidence
    print(f"  Intent: {decision.intent.value} (confidence: {conf:.2f})")

    # Step 2: Route to appropriate handler
    handler = INTENT_HANDLERS[decision.intent]
    print(f"  Routing to: {decision.intent.value} handler")

    handler_result = await handler.run(
        f"Handle this customer query:\n\n{user_query}"
    )

    print("  Handler complete.")
    return decision, handler_result.output


# --8<-- [end:routing]


if __name__ == "__main__":
    import asyncio

    test_queries = [
        "Where is my order ORD-12345? It was supposed to arrive yesterday.",
        "What features does the Pro model have compared to the Basic?",
        "My device won't turn on after the latest update.",
        "Hello, I need help with something but I'm not sure what.",
    ]

    async def main() -> None:
        for query in test_queries:
            print("\n" + "=" * 60)
            print(f"Query: {query}")
            print("=" * 60)

            decision, response = await route_query(query)

            print("\nRouting Decision:")
            print(f"  Intent: {decision.intent.value}")
            print(f"  Confidence: {decision.confidence:.2f}")
            print(f"  Reasoning: {decision.reasoning}")

            print(f"\nResponse ({type(response).__name__}):")
            for field, value in response.model_dump().items():
                print(f"  {field}: {value}")

    asyncio.run(main())
