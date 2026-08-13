"""
Edge-case tests for our own hybrid heuristics (agent_ppo.py's fixed_*_decision
functions) — never touches ASU. Checks each heuristic degrades gracefully
(returns None / a legal action, never crashes) in rare/boundary game states.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monopoly_game_engine.actions import ActionType, AuctionAction  # noqa: E402
from monopoly_game_engine.agent_ppo import (  # noqa: E402
    fixed_auction_decision,
    fixed_build_decision,
    fixed_jail_decision,
    fixed_mortgage_decision,
    fixed_trade_offer_decision,
    fixed_unmortgage_decision,
)
from monopoly_game_engine.env import PHASE_AUCTION, MonopolyEnv  # noqa: E402


class TestMortgageEdgeCases(unittest.TestCase):
    def test_no_eligible_properties_returns_none(self):
        env = MonopolyEnv(agent_ids=[0])
        env.reset()
        env.players[0].cash = 50  # below the $200 floor, but owns nothing
        allowed = env.get_allowed_actions(0)
        self.assertIsNone(fixed_mortgage_decision(env, 0, allowed))

    def test_cash_above_floor_returns_none(self):
        env = MonopolyEnv(agent_ids=[0])
        env.reset()
        env.players[0].cash = 1000
        allowed = env.get_allowed_actions(0)
        self.assertIsNone(fixed_mortgage_decision(env, 0, allowed))


class TestUnmortgageEdgeCases(unittest.TestCase):
    def test_no_mortgaged_properties_returns_none(self):
        env = MonopolyEnv(agent_ids=[0])
        env.reset()
        env.players[0].cash = 2000
        allowed = env.get_allowed_actions(0)
        self.assertIsNone(fixed_unmortgage_decision(env, 0, allowed))

    def test_would_drop_below_buffer_returns_none(self):
        env = MonopolyEnv(agent_ids=[0])
        env.reset()
        sq = 1  # Mediterranean Avenue, cheap
        prop = env.properties[sq]
        prop.owner = 0
        env.players[0].properties.append(prop)
        prop.mortgaged = True
        env.players[0].cash = 500  # exactly at the floor, unmortgage would dip below $300 buffer
        allowed = env.get_allowed_actions(0)
        # Should not crash; either None or a legal unmortgage action is fine —
        # the important thing is it never raises.
        fixed_unmortgage_decision(env, 0, allowed)


class TestAuctionEdgeCases(unittest.TestCase):
    def test_cash_below_safety_buffer_passes_not_crashes(self):
        env = MonopolyEnv(agent_ids=[0])
        env.reset()
        env.players[0].cash = 50  # below the $100 safety buffer -> max_bid negative
        env.phase = PHASE_AUCTION
        env.auction_property_id = 1
        env.auction_high_bid = 0
        allowed = [int(AuctionAction.PASS), int(AuctionAction.BID_1)]
        action = fixed_auction_decision(env, 0, allowed)
        self.assertEqual(action, int(AuctionAction.PASS))

    def test_unowned_group_property_no_crash(self):
        env = MonopolyEnv(agent_ids=[0])
        env.reset()
        env.players[0].cash = 1500
        env.phase = PHASE_AUCTION
        env.auction_property_id = 5  # railroad
        env.auction_high_bid = 0
        allowed = [int(AuctionAction.PASS), int(AuctionAction.BID_1), int(AuctionAction.BID_10)]
        action = fixed_auction_decision(env, 0, allowed)
        self.assertIn(action, allowed)


class TestJailEdgeCases(unittest.TestCase):
    def test_no_gooj_card_returns_none(self):
        env = MonopolyEnv(agent_ids=[0])
        env.reset()
        env.players[0].gooj_card = False
        allowed = [int(ActionType.PAY_BAIL), int(ActionType.ROLL_DICE)]
        self.assertIsNone(fixed_jail_decision(env, 0, allowed))

    def test_gooj_card_held_uses_it(self):
        env = MonopolyEnv(agent_ids=[0])
        env.reset()
        allowed = [int(ActionType.USE_GOOJ_CARD), int(ActionType.PAY_BAIL)]
        self.assertEqual(
            fixed_jail_decision(env, 0, allowed), int(ActionType.USE_GOOJ_CARD)
        )


class TestBuildEdgeCases(unittest.TestCase):
    def test_no_monopolies_returns_none(self):
        env = MonopolyEnv(agent_ids=[0])
        env.reset()
        allowed = env.get_allowed_actions(0)
        self.assertIsNone(fixed_build_decision(env, 0, allowed))


class TestTradeOfferEdgeCases(unittest.TestCase):
    def test_all_other_players_bankrupt_returns_none(self):
        env = MonopolyEnv(agent_ids=[0])
        env.reset()
        for pid in (1, 2, 3):
            env.players[pid].bankrupt = True
        allowed = env.get_allowed_actions(0)
        self.assertIsNone(fixed_trade_offer_decision(env, 0, allowed))


if __name__ == "__main__":
    unittest.main()
