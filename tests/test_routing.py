"""Tests for the Routing Pattern implementation."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from agentic_patterns.routing import INTENT_HANDLERS
from agentic_patterns.routing import ClarificationResponse
from agentic_patterns.routing import Intent
from agentic_patterns.routing import OrderStatusResponse
from agentic_patterns.routing import ProductInfoResponse
from agentic_patterns.routing import RouteDecision
from agentic_patterns.routing import TechnicalSupportResponse
from agentic_patterns.routing import route_query


class TestModels:
    """Test Pydantic model validation."""

    def test_intent_enum_values(self):
        assert Intent.ORDER_STATUS.value == "order_status"
        assert Intent.PRODUCT_INFO.value == "product_info"
        assert Intent.TECHNICAL_SUPPORT.value == "technical_support"
        assert Intent.CLARIFICATION.value == "clarification"

    def test_route_decision_valid(self):
        decision = RouteDecision(
            intent=Intent.ORDER_STATUS,
            confidence=0.95,
            reasoning="User asked about order tracking",
        )
        assert decision.intent == Intent.ORDER_STATUS
        assert decision.confidence == 0.95

    def test_route_decision_confidence_bounds(self):
        decision_low = RouteDecision(
            intent=Intent.PRODUCT_INFO,
            confidence=0.0,
            reasoning="Very uncertain",
        )
        assert decision_low.confidence == 0.0

        decision_high = RouteDecision(
            intent=Intent.PRODUCT_INFO,
            confidence=1.0,
            reasoning="Very certain",
        )
        assert decision_high.confidence == 1.0

    def test_route_decision_invalid_confidence(self):
        with pytest.raises(ValueError):
            RouteDecision(
                intent=Intent.ORDER_STATUS,
                confidence=1.5,
                reasoning="Invalid",
            )

        with pytest.raises(ValueError):
            RouteDecision(
                intent=Intent.ORDER_STATUS,
                confidence=-0.1,
                reasoning="Invalid",
            )

    def test_order_status_response_valid(self):
        response = OrderStatusResponse(
            order_id="ORD-12345",
            status_message="Your order is in transit",
            next_steps=["Track on website", "Contact support if delayed"],
        )
        assert response.order_id == "ORD-12345"
        assert len(response.next_steps) == 2

    def test_order_status_response_optional_order_id(self):
        response = OrderStatusResponse(
            status_message="Please provide your order ID",
        )
        assert response.order_id is None
        assert len(response.next_steps) == 0

    def test_product_info_response_valid(self):
        response = ProductInfoResponse(
            product_name="Widget Pro",
            information="The Widget Pro has advanced features...",
            related_products=["Widget Basic", "Widget Ultra"],
        )
        assert response.product_name == "Widget Pro"
        assert len(response.related_products) == 2

    def test_product_info_response_optional_fields(self):
        response = ProductInfoResponse(
            information="General product catalog information",
        )
        assert response.product_name is None
        assert len(response.related_products) == 0

    def test_technical_support_response_valid(self):
        response = TechnicalSupportResponse(
            issue_category="Software Update",
            troubleshooting_steps=["Restart device", "Check for updates"],
            escalation_needed=False,
        )
        assert response.issue_category == "Software Update"
        assert not response.escalation_needed

    def test_technical_support_escalation(self):
        response = TechnicalSupportResponse(
            issue_category="Hardware Failure",
            troubleshooting_steps=["Contact support"],
            escalation_needed=True,
        )
        assert response.escalation_needed

    def test_clarification_response_valid(self):
        response = ClarificationResponse(
            original_query="Help",
            clarifying_questions=["What do you need help with?"],
            possible_intents=["Order status", "Product info", "Support"],
        )
        assert response.original_query == "Help"
        assert len(response.clarifying_questions) == 1
        assert len(response.possible_intents) == 3


class TestIntentHandlers:
    """Test that intent handlers are properly mapped."""

    def test_all_intents_have_handlers(self):
        for intent in Intent:
            assert intent in INTENT_HANDLERS

    def test_handler_count_matches_intent_count(self):
        assert len(INTENT_HANDLERS) == len(Intent)


class TestRouteQuery:
    """Test the routing flow with mocked agents."""

    @pytest.fixture
    def mock_route_decision_order(self):
        return RouteDecision(
            intent=Intent.ORDER_STATUS,
            confidence=0.92,
            reasoning="User mentioned order tracking",
        )

    @pytest.fixture
    def mock_route_decision_product(self):
        return RouteDecision(
            intent=Intent.PRODUCT_INFO,
            confidence=0.88,
            reasoning="User asking about product features",
        )

    @pytest.fixture
    def mock_route_decision_support(self):
        return RouteDecision(
            intent=Intent.TECHNICAL_SUPPORT,
            confidence=0.85,
            reasoning="User reporting technical issue",
        )

    @pytest.fixture
    def mock_route_decision_clarify(self):
        return RouteDecision(
            intent=Intent.CLARIFICATION,
            confidence=0.45,
            reasoning="Query is unclear",
        )

    @pytest.fixture
    def mock_order_response(self):
        return OrderStatusResponse(
            order_id="ORD-12345",
            status_message="Your order is being processed",
            next_steps=["Check email for updates"],
        )

    @pytest.fixture
    def mock_product_response(self):
        return ProductInfoResponse(
            product_name="Pro Model",
            information="The Pro model includes advanced features",
            related_products=["Basic Model"],
        )

    @pytest.fixture
    def mock_support_response(self):
        return TechnicalSupportResponse(
            issue_category="Device Power",
            troubleshooting_steps=["Hold power button", "Check battery"],
            escalation_needed=False,
        )

    @pytest.fixture
    def mock_clarify_response(self):
        return ClarificationResponse(
            original_query="Help me",
            clarifying_questions=["What do you need help with?"],
            possible_intents=["order_status", "product_info"],
        )

    @pytest.mark.asyncio
    async def test_route_to_order_status(
        self, mock_route_decision_order, mock_order_response
    ):
        """Test routing to order status handler."""
        mock_router_result = MagicMock()
        mock_router_result.output = mock_route_decision_order

        mock_handler_result = MagicMock()
        mock_handler_result.output = mock_order_response

        mock_handler = MagicMock()
        mock_handler.run = AsyncMock(return_value=mock_handler_result)

        with (
            patch("agentic_patterns.routing.router_agent") as mock_router,
            patch.dict(
                "agentic_patterns.routing.INTENT_HANDLERS",
                {Intent.ORDER_STATUS: mock_handler},
            ),
        ):
            mock_router.run = AsyncMock(return_value=mock_router_result)

            decision, response = await route_query(
                "Where is my order ORD-12345?"
            )

            assert decision.intent == Intent.ORDER_STATUS
            assert isinstance(response, OrderStatusResponse)
            assert response.order_id == "ORD-12345"
            mock_router.run.assert_called_once()
            mock_handler.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_product_info(
        self, mock_route_decision_product, mock_product_response
    ):
        """Test routing to product info handler."""
        mock_router_result = MagicMock()
        mock_router_result.output = mock_route_decision_product

        mock_handler_result = MagicMock()
        mock_handler_result.output = mock_product_response

        mock_handler = MagicMock()
        mock_handler.run = AsyncMock(return_value=mock_handler_result)

        with (
            patch("agentic_patterns.routing.router_agent") as mock_router,
            patch.dict(
                "agentic_patterns.routing.INTENT_HANDLERS",
                {Intent.PRODUCT_INFO: mock_handler},
            ),
        ):
            mock_router.run = AsyncMock(return_value=mock_router_result)

            decision, response = await route_query(
                "What features does the Pro model have?"
            )

            assert decision.intent == Intent.PRODUCT_INFO
            assert isinstance(response, ProductInfoResponse)
            assert response.product_name == "Pro Model"

    @pytest.mark.asyncio
    async def test_route_to_technical_support(
        self, mock_route_decision_support, mock_support_response
    ):
        """Test routing to technical support handler."""
        mock_router_result = MagicMock()
        mock_router_result.output = mock_route_decision_support

        mock_handler_result = MagicMock()
        mock_handler_result.output = mock_support_response

        mock_handler = MagicMock()
        mock_handler.run = AsyncMock(return_value=mock_handler_result)

        with (
            patch("agentic_patterns.routing.router_agent") as mock_router,
            patch.dict(
                "agentic_patterns.routing.INTENT_HANDLERS",
                {Intent.TECHNICAL_SUPPORT: mock_handler},
            ),
        ):
            mock_router.run = AsyncMock(return_value=mock_router_result)

            decision, response = await route_query("My device won't turn on")

            assert decision.intent == Intent.TECHNICAL_SUPPORT
            assert isinstance(response, TechnicalSupportResponse)
            assert not response.escalation_needed

    @pytest.mark.asyncio
    async def test_route_to_clarification(
        self, mock_route_decision_clarify, mock_clarify_response
    ):
        """Test routing to clarification handler for unclear queries."""
        mock_router_result = MagicMock()
        mock_router_result.output = mock_route_decision_clarify

        mock_handler_result = MagicMock()
        mock_handler_result.output = mock_clarify_response

        mock_handler = MagicMock()
        mock_handler.run = AsyncMock(return_value=mock_handler_result)

        with (
            patch("agentic_patterns.routing.router_agent") as mock_router,
            patch.dict(
                "agentic_patterns.routing.INTENT_HANDLERS",
                {Intent.CLARIFICATION: mock_handler},
            ),
        ):
            mock_router.run = AsyncMock(return_value=mock_router_result)

            decision, response = await route_query("Help me")

            assert decision.intent == Intent.CLARIFICATION
            assert isinstance(response, ClarificationResponse)
            assert len(response.clarifying_questions) > 0

    @pytest.mark.asyncio
    async def test_query_passed_to_handler(
        self, mock_route_decision_order, mock_order_response
    ):
        """Verify the original query is passed to the handler."""
        mock_router_result = MagicMock()
        mock_router_result.output = mock_route_decision_order

        mock_handler_result = MagicMock()
        mock_handler_result.output = mock_order_response

        mock_handler = MagicMock()
        mock_handler.run = AsyncMock(return_value=mock_handler_result)

        test_query = "Where is my order ORD-99999?"

        with (
            patch("agentic_patterns.routing.router_agent") as mock_router,
            patch.dict(
                "agentic_patterns.routing.INTENT_HANDLERS",
                {Intent.ORDER_STATUS: mock_handler},
            ),
        ):
            mock_router.run = AsyncMock(return_value=mock_router_result)

            await route_query(test_query)

            handler_call_args = mock_handler.run.call_args[0][0]
            assert "ORD-99999" in handler_call_args


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_next_steps(self):
        response = OrderStatusResponse(
            status_message="Status unknown",
            next_steps=[],
        )
        assert len(response.next_steps) == 0

    def test_empty_troubleshooting_steps(self):
        response = TechnicalSupportResponse(
            issue_category="Unknown",
            troubleshooting_steps=[],
            escalation_needed=True,
        )
        assert len(response.troubleshooting_steps) == 0
        assert response.escalation_needed

    def test_empty_clarifying_questions(self):
        response = ClarificationResponse(
            original_query="?",
            clarifying_questions=[],
            possible_intents=[],
        )
        assert len(response.clarifying_questions) == 0
        assert len(response.possible_intents) == 0
