"""Tests for _middleware.py — MiddlewareData, register_middleware, _run_chain."""

import pytest

from orkestra._middleware import (
    MiddlewareData,
    _global_middlewares,
    _run_chain,
    register_middleware,
)


# ---------------------------------------------------------------------------
# Isolation — reset global middleware list between every test in this file
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_global_middlewares():
    original = list(_global_middlewares)
    _global_middlewares.clear()
    yield
    _global_middlewares.clear()
    _global_middlewares.extend(original)


# ---------------------------------------------------------------------------
# MiddlewareData
# ---------------------------------------------------------------------------

class TestMiddlewareData:
    def test_required_fields(self):
        data = MiddlewareData(
            prompt="hello",
            provider="google",
            model="gemini-3-flash-preview",
            max_tokens=512,
            temperature=0.7,
            event="chat",
        )
        assert data.prompt == "hello"
        assert data.provider == "google"
        assert data.model == "gemini-3-flash-preview"
        assert data.max_tokens == 512
        assert data.temperature == 0.7
        assert data.event == "chat"

    def test_response_defaults_none(self):
        data = MiddlewareData(
            prompt="hi", provider="p", model=None,
            max_tokens=8192, temperature=1.0, event="chat"
        )
        assert data.response is None

    def test_metadata_defaults_empty_dict(self):
        data = MiddlewareData(
            prompt="hi", provider="p", model=None,
            max_tokens=8192, temperature=1.0, event="chat"
        )
        assert data.metadata == {}

    def test_metadata_is_independent_per_instance(self):
        a = MiddlewareData(prompt="hi", provider="p", model=None, max_tokens=8, temperature=1.0, event="chat")
        b = MiddlewareData(prompt="hi", provider="p", model=None, max_tokens=8, temperature=1.0, event="chat")
        a.metadata["x"] = 1
        assert "x" not in b.metadata

    def test_fields_are_mutable(self):
        data = MiddlewareData(prompt="original", provider="p", model=None, max_tokens=8, temperature=1.0, event="chat")
        data.prompt = "modified"
        data.model = "claude-haiku-4"
        data.response = "fake_response"
        assert data.prompt == "modified"
        assert data.model == "claude-haiku-4"
        assert data.response == "fake_response"


# ---------------------------------------------------------------------------
# register_middleware
# ---------------------------------------------------------------------------

class TestRegisterMiddleware:
    def test_decorator_style_registers_fn(self):
        @register_middleware
        def my_mw(data, next):
            next()

        assert my_mw in _global_middlewares

    def test_functional_style_registers_fn(self):
        def my_mw(data, next):
            next()

        register_middleware(my_mw)
        assert my_mw in _global_middlewares

    def test_returns_original_function_unchanged(self):
        def my_mw(data, next):
            next()

        result = register_middleware(my_mw)
        assert result is my_mw

    def test_multiple_registrations_all_present(self):
        def mw1(data, next): next()
        def mw2(data, next): next()

        register_middleware(mw1)
        register_middleware(mw2)
        assert mw1 in _global_middlewares
        assert mw2 in _global_middlewares

    def test_registration_order_preserved(self):
        def mw1(data, next): next()
        def mw2(data, next): next()
        def mw3(data, next): next()

        register_middleware(mw1)
        register_middleware(mw2)
        register_middleware(mw3)
        idx = {fn: _global_middlewares.index(fn) for fn in [mw1, mw2, mw3]}
        assert idx[mw1] < idx[mw2] < idx[mw3]


# ---------------------------------------------------------------------------
# _run_chain
# ---------------------------------------------------------------------------

def _make_data(**kwargs):
    defaults = dict(prompt="hi", provider="p", model=None, max_tokens=8, temperature=1.0, event="chat")
    defaults.update(kwargs)
    return MiddlewareData(**defaults)


class TestRunChainNoMiddleware:
    def test_empty_list_calls_final_directly(self):
        called = []
        data = _make_data()
        _run_chain([], data, lambda d: called.append("final"))
        assert called == ["final"]

    def test_final_receives_same_data_object(self):
        received = []
        data = _make_data()
        _run_chain([], data, lambda d: received.append(d))
        assert received[0] is data


class TestRunChainSingleMiddleware:
    def test_middleware_called_before_final(self):
        order = []
        data = _make_data()

        def mw(d, nxt):
            order.append("mw")
            nxt()

        _run_chain([mw], data, lambda d: order.append("final"))
        assert order == ["mw", "final"]

    def test_middleware_not_calling_next_skips_final(self):
        called = []
        data = _make_data()

        def mw(d, nxt):
            called.append("mw")
            # deliberately skip nxt()

        _run_chain([mw], data, lambda d: called.append("final"))
        assert called == ["mw"]

    def test_middleware_receives_data(self):
        received = []
        data = _make_data(prompt="test_prompt")

        def mw(d, nxt):
            received.append(d.prompt)
            nxt()

        _run_chain([mw], data, lambda d: None)
        assert received == ["test_prompt"]

    def test_middleware_and_final_share_same_data_object(self):
        objects = []
        data = _make_data()

        def mw(d, nxt):
            objects.append(d)
            nxt()

        _run_chain([mw], data, lambda d: objects.append(d))
        assert objects[0] is objects[1] is data


class TestRunChainMultipleMiddlewares:
    def test_called_in_fifo_order(self):
        order = []
        data = _make_data()

        def mw1(d, nxt): order.append(1); nxt()
        def mw2(d, nxt): order.append(2); nxt()
        def mw3(d, nxt): order.append(3); nxt()

        _run_chain([mw1, mw2, mw3], data, lambda d: order.append("final"))
        assert order == [1, 2, 3, "final"]

    def test_mutations_visible_to_downstream(self):
        data = _make_data(prompt="original")

        def mw1(d, nxt):
            d.prompt = "mutated_by_mw1"
            nxt()

        def mw2(d, nxt):
            d.prompt = d.prompt + "_and_mw2"
            nxt()

        _run_chain([mw1, mw2], data, lambda d: None)
        assert data.prompt == "mutated_by_mw1_and_mw2"

    def test_final_sees_mutations_from_all_middlewares(self):
        data = _make_data()
        seen_in_final = []

        def mw1(d, nxt): d.metadata["mw1"] = True; nxt()
        def mw2(d, nxt): d.metadata["mw2"] = True; nxt()

        _run_chain([mw1, mw2], data, lambda d: seen_in_final.append(dict(d.metadata)))
        assert seen_in_final[0] == {"mw1": True, "mw2": True}

    def test_middleware_can_read_response_after_next(self):
        data = _make_data()
        post_next_values = []

        def mw(d, nxt):
            nxt()
            post_next_values.append(d.response)

        def final(d):
            d.response = "fake_response"

        _run_chain([mw], data, final)
        assert post_next_values == ["fake_response"]

    def test_short_circuit_in_middle_skips_rest(self):
        order = []
        data = _make_data()

        def mw1(d, nxt): order.append(1); nxt()
        def mw2(d, nxt): order.append(2)  # no next()
        def mw3(d, nxt): order.append(3); nxt()

        _run_chain([mw1, mw2, mw3], data, lambda d: order.append("final"))
        assert order == [1, 2]

    def test_first_middleware_short_circuit_skips_all(self):
        order = []
        data = _make_data()

        def mw1(d, nxt): order.append(1)  # no next()
        def mw2(d, nxt): order.append(2); nxt()

        _run_chain([mw1, mw2], data, lambda d: order.append("final"))
        assert order == [1]

    def test_post_next_code_runs_in_reverse_order(self):
        """Middleware post-next code runs in reverse (innermost first)."""
        post = []
        data = _make_data()

        def mw1(d, nxt): nxt(); post.append("post_mw1")
        def mw2(d, nxt): nxt(); post.append("post_mw2")

        _run_chain([mw1, mw2], data, lambda d: None)
        assert post == ["post_mw2", "post_mw1"]

    def test_middleware_can_set_response_on_short_circuit(self):
        data = _make_data()

        def mw(d, nxt):
            d.response = "short_circuit_response"
            # does not call nxt()

        _run_chain([mw], data, lambda d: d.__setattr__("response", "final_response"))
        assert data.response == "short_circuit_response"

    def test_all_middlewares_share_same_data_object(self):
        ids = []
        data = _make_data()

        def mw1(d, nxt): ids.append(id(d)); nxt()
        def mw2(d, nxt): ids.append(id(d)); nxt()

        _run_chain([mw1, mw2], data, lambda d: ids.append(id(d)))
        assert len(set(ids)) == 1  # all the same object
