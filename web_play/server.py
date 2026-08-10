"""
web_play/server.py
-------------------
Local Flask backend that lets a human play against the repo's fixed-policy
bots through the real MonopolyEnv (monopoly_game_engine). No RL model
required - bots use the deterministic FP-A..F personalities.

Run:
    python web_play/server.py
Then open http://127.0.0.1:5000
"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request, send_from_directory

from monopoly_game_engine.actions import ActionType, AuctionAction, OFFSETS
from monopoly_game_engine.agents_fixed import FP_AGENT_CLASSES
from monopoly_game_engine.constants import (
    BOARD,
    GO_TO_JAIL_SQUARE,
    INCOME_TAX_SQUARE,
    JAIL_BAIL,
    JAIL_SQUARE,
    LUXURY_TAX_SQUARE,
    PROPERTIES,
    PROPERTY_IDS,
    REAL_ESTATE_IDS,
    TRADE_CASH_LEVELS,
)
from monopoly_game_engine.env import PHASE_AUCTION, PHASE_OUT_OF_TURN, MonopolyEnv

app = Flask(__name__, static_folder="static", static_url_path="")

HUMAN_PID = 0
MAX_BOT_STEPS_PER_CALL = 2000

SESSION = {"env": None, "bots": {}, "pnames": {}, "log": [], "n_players": 4}


# ── Action label prettifier (Turkish, human-facing) ────────────────────────


def _prop_name(sq):
    return PROPERTIES[sq]["name"]


def prettify_action(action_idx, env, pid):
    n_players = len(env.players)
    others = [i for i in range(n_players) if i != pid]

    if action_idx < OFFSETS["mortgage"]:
        atype = ActionType(action_idx)
        if atype == ActionType.DO_NOTHING:
            return "Bekle"
        if atype == ActionType.END_TURN:
            return "Turu bitir"
        if atype == ActionType.ROLL_DICE:
            return "Zar at"
        if atype == ActionType.BUY_PROPERTY:
            sq = env.players[pid].position
            return f"Satın al — {_prop_name(sq)} (${env.properties[sq].price})"
        if atype == ActionType.USE_GOOJ_CARD:
            return "Hapisten Çıkış Kartı kullan"
        if atype == ActionType.PAY_BAIL:
            return f"Kefalet öde (${JAIL_BAIL})"
        if atype == ActionType.DECLARE_BANKRUPT:
            return "İflasını ilan et"
        if atype == ActionType.ACCEPT_TRADE:
            return "Teklifi kabul et"
        if atype == ActionType.DECLINE_TRADE:
            return "Teklifi reddet"

    if action_idx < OFFSETS["unmortgage"]:
        sq = PROPERTY_IDS[action_idx - OFFSETS["mortgage"]]
        prop = env.properties[sq]
        return f"İpotek ver — {prop.name} (+${prop.mortgage_v})"

    if action_idx < OFFSETS["improve_house"]:
        sq = PROPERTY_IDS[action_idx - OFFSETS["unmortgage"]]
        prop = env.properties[sq]
        cost = int(prop.mortgage_v * 1.1)
        return f"İpoteği kaldır — {prop.name} (-${cost})"

    if action_idx < OFFSETS["improve_hotel"]:
        sq = REAL_ESTATE_IDS[action_idx - OFFSETS["improve_house"]]
        prop = env.properties[sq]
        return f"Ev inşa et — {prop.name} (-${prop.data['house_price']})"

    if action_idx < OFFSETS["sell_house"]:
        sq = REAL_ESTATE_IDS[action_idx - OFFSETS["improve_hotel"]]
        prop = env.properties[sq]
        return f"Otel inşa et — {prop.name} (-${prop.data['house_price']})"

    if action_idx < OFFSETS["sell_hotel"]:
        sq = REAL_ESTATE_IDS[action_idx - OFFSETS["sell_house"]]
        prop = env.properties[sq]
        return f"Ev sat — {prop.name} (+${prop.data['house_price'] // 2})"

    if action_idx < OFFSETS["sell_prop"]:
        sq = REAL_ESTATE_IDS[action_idx - OFFSETS["sell_hotel"]]
        prop = env.properties[sq]
        return f"Otel sat — {prop.name} (+${prop.data['house_price'] // 2})"

    if action_idx < OFFSETS["buy_trade"]:
        sq = PROPERTY_IDS[action_idx - OFFSETS["sell_prop"]]
        prop = env.properties[sq]
        return f"Bankaya sat — {prop.name} (+${prop.mortgage_v})"

    n = len(PROPERTY_IDS)
    nc = len(TRADE_CASH_LEVELS)

    if action_idx < OFFSETS["sell_trade"]:
        local = action_idx - OFFSETS["buy_trade"]
        t_idx, rem = divmod(local, n * nc)
        prop_idx, price_idx = divmod(rem, nc)
        target = others[t_idx] if t_idx < len(others) else others[0]
        sq = PROPERTY_IDS[prop_idx]
        cash = int(env.properties[sq].price * TRADE_CASH_LEVELS[price_idx])
        return f"Teklif gönder — {SESSION['pnames'].get(target, f'Oyuncu {target+1}')}'a ${cash} karşılığında {_prop_name(sq)} iste"

    if action_idx < OFFSETS["exch_trade"]:
        local = action_idx - OFFSETS["sell_trade"]
        t_idx, rem = divmod(local, n * nc)
        prop_idx, price_idx = divmod(rem, nc)
        target = others[t_idx] if t_idx < len(others) else others[0]
        sq = PROPERTY_IDS[prop_idx]
        cash = int(env.properties[sq].price * TRADE_CASH_LEVELS[price_idx])
        return f"Teklif gönder — {SESSION['pnames'].get(target, f'Oyuncu {target+1}')}'a ${cash} karşılığında {_prop_name(sq)} sat"

    if action_idx < OFFSETS["auction"]:
        local = action_idx - OFFSETS["exch_trade"]
        t_idx, rem = divmod(local, n * (n - 1))
        offer_idx, req_raw = divmod(rem, n - 1)
        req_idx = req_raw if req_raw < offer_idx else req_raw + 1
        target = others[t_idx] if t_idx < len(others) else others[0]
        return (
            f"Takas teklif et — {SESSION['pnames'].get(target, f'Oyuncu {target+1}')}'a "
            f"{_prop_name(PROPERTY_IDS[offer_idx])} ver, {_prop_name(PROPERTY_IDS[req_idx])} al"
        )

    # Auction bidding
    if action_idx == int(AuctionAction.PASS):
        return "Açık artırmayı geç"
    from monopoly_game_engine.actions import AUCTION_ACTION_TO_INCREMENT

    increment = AUCTION_ACTION_TO_INCREMENT[AuctionAction(action_idx)]
    return f"Teklif ver (+${increment})"


# ── Log helpers ─────────────────────────────────────────────────────────────


def log(msg):
    SESSION["log"].append(msg)


def square_name(sq):
    return BOARD.get(sq, f"Kare {sq}")


def log_step(pid, action_idx, env, info):
    pname = SESSION["pnames"].get(pid, f"Oyuncu {pid+1}")
    atype = None
    if action_idx < OFFSETS["mortgage"]:
        atype = ActionType(action_idx)

    if atype == ActionType.ROLL_DICE and "dice" in info:
        d1, d2 = info["dice"]
        log(f"{pname}: {d1} ve {d2} attı (toplam {d1+d2})")
        sq = env.players[pid].position
        if sq == GO_TO_JAIL_SQUARE:
            log(f"  → Doğrudan hapse gönderildi")
        elif sq == JAIL_SQUARE and env.players[pid].in_jail:
            log(f"  → Hapiste")
        elif sq == INCOME_TAX_SQUARE:
            log(f"  → Gelir Vergisi karesi, $200 ödedi")
        elif sq == LUXURY_TAX_SQUARE:
            log(f"  → Lüks Vergi karesi, $100 ödedi")
        elif sq in env.properties:
            prop = env.properties[sq]
            if prop.owner is None:
                log(f"  → {prop.name} karesine geldi (boş, ${prop.price})")
            elif prop.owner == pid:
                log(f"  → kendi mülkü {prop.name} karesine geldi")
            else:
                rent = info.get("rent_paid", "?")
                owner_pn = SESSION["pnames"].get(prop.owner, f"Oyuncu {prop.owner+1}")
                log(f"  → {owner_pn}'e ait {prop.name} karesine geldi, ${rent} kira ödedi")
        else:
            log(f"  → {square_name(sq)} karesine geldi")
    elif atype == ActionType.BUY_PROPERTY:
        sq = env.players[pid].position
        prop = env.properties.get(sq)
        if prop:
            log(f"{pname}: {prop.name} satın aldı (${prop.price})")
    elif atype == ActionType.DECLARE_BANKRUPT:
        log(f"💀 {pname} İFLAS ETTİ")
    elif atype not in (ActionType.DO_NOTHING, None):
        log(f"{pname}: {prettify_action(action_idx, env, pid)}")
    elif atype is None:
        log(f"{pname}: {prettify_action(action_idx, env, pid)}")

    if "auction_winner" in info:
        winner_pn = SESSION["pnames"].get(info["auction_winner"], f"Oyuncu {info['auction_winner']+1}")
        log(f"  → Açık artırmayı {winner_pn} kazandı (${info['auction_price']})")


# ── Game session management ─────────────────────────────────────────────────


def new_game(n_players=4, seed=None):
    if seed is not None:
        random.seed(seed)

    env = MonopolyEnv(agent_ids=[HUMAN_PID], max_rounds=200)
    env.reset()

    pnames = {HUMAN_PID: "Sen"}
    bot_classes = random.sample(FP_AGENT_CLASSES, k=min(len(FP_AGENT_CLASSES), n_players - 1))
    bots = {}
    other_pids = list(range(1, n_players))
    for i, pid in enumerate(other_pids):
        cls = bot_classes[i % len(bot_classes)]
        bots[pid] = cls(pid)
        pnames[pid] = f"{cls.__name__} ({pid+1})"

    for pid in range(n_players, 4):
        env.players[pid].bankrupt = True

    env.turn_order = [p for p in env.turn_order if p < n_players]
    env.current_turn_idx = 0

    SESSION["env"] = env
    SESSION["bots"] = bots
    SESSION["pnames"] = pnames
    SESSION["log"] = []
    SESSION["n_players"] = n_players

    log(f"Yeni oyun — {n_players} oyuncu")
    log(f"Sıra: {' → '.join(pnames[p] for p in env.turn_order)}")

    run_bots()


def run_bots():
    env = SESSION["env"]
    bots = SESSION["bots"]
    steps = 0
    while not env.done and env.whose_turn() != HUMAN_PID and steps < MAX_BOT_STEPS_PER_CALL:
        steps += 1
        pid = env.whose_turn()
        if env.players[pid].bankrupt:
            env._advance_turn()
            continue
        allowed = env.get_allowed_actions(pid)
        if not allowed:
            allowed = [int(ActionType.END_TURN)]
        agent = bots.get(pid)
        action = agent.choose_action(env) if agent else int(ActionType.END_TURN)
        if action not in allowed:
            action = int(ActionType.END_TURN) if int(ActionType.END_TURN) in allowed else allowed[0]
        _, _, done, info = env.step(action)
        if action != int(ActionType.DO_NOTHING):
            log_step(pid, action, env, info)


def state_json():
    env = SESSION["env"]
    if env is None:
        return {"started": False}

    players = []
    for pid, p in enumerate(env.players[: SESSION["n_players"]]):
        players.append(
            {
                "pid": pid,
                "name": SESSION["pnames"].get(pid, f"Oyuncu {pid+1}"),
                "cash": p.cash,
                "position": p.position,
                "in_jail": p.in_jail,
                "gooj_card": p.gooj_card,
                "bankrupt": p.bankrupt,
                "net_worth": round(p.net_worth()),
                "properties": [pr.square_id for pr in p.properties],
                "is_human": pid == HUMAN_PID,
            }
        )

    properties = {}
    for sq, prop in env.properties.items():
        properties[sq] = {
            "owner": prop.owner,
            "mortgaged": prop.mortgaged,
            "houses": prop.houses,
        }

    whose_turn = env.whose_turn() if not env.done else None
    allowed_actions = []
    if not env.done and whose_turn == HUMAN_PID:
        for a in env.get_allowed_actions(HUMAN_PID):
            allowed_actions.append({"id": a, "label": prettify_action(a, env, HUMAN_PID)})

    incoming_trade = None
    offer = env._incoming_trade(HUMAN_PID)
    if offer is not None:
        incoming_trade = {
            "from": SESSION["pnames"].get(offer.from_player, f"Oyuncu {offer.from_player+1}"),
            "offered_prop": offer.offered_prop.name if offer.offered_prop else None,
            "requested_prop": offer.requested_prop.name if offer.requested_prop else None,
            "cash_offered": offer.cash_offered,
            "cash_requested": offer.cash_requested,
        }

    auction = None
    if env.phase == PHASE_AUCTION:
        auction = {
            "property": env.properties[env.auction_property_id].name if env.auction_property_id else None,
            "high_bid": env.auction_high_bid,
            "high_bidder": SESSION["pnames"].get(env.auction_high_bidder) if env.auction_high_bidder is not None else None,
            "current_pid": env.auction_current_pid,
        }

    winner = None
    if env.done:
        w = env.winner()
        winner = SESSION["pnames"].get(w, f"Oyuncu {w+1}")

    return {
        "started": True,
        "round": env.round,
        "phase": env.phase,
        "done": env.done,
        "winner": winner,
        "whose_turn": whose_turn,
        "whose_turn_name": SESSION["pnames"].get(whose_turn) if whose_turn is not None else None,
        "human_pid": HUMAN_PID,
        "last_dice": list(env.last_dice),
        "players": players,
        "properties": properties,
        "allowed_actions": allowed_actions,
        "incoming_trade": incoming_trade,
        "auction": auction,
        "log": SESSION["log"][-60:],
    }


# ── HTTP routes ──────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/state")
def api_state():
    return jsonify(state_json())


@app.route("/api/reset", methods=["POST"])
def api_reset():
    body = request.get_json(silent=True) or {}
    n_players = int(body.get("players", 4))
    n_players = max(2, min(4, n_players))
    new_game(n_players=n_players)
    return jsonify(state_json())


@app.route("/api/action", methods=["POST"])
def api_action():
    env = SESSION["env"]
    if env is None:
        return jsonify({"error": "no active game"}), 400
    if env.done:
        return jsonify(state_json())

    body = request.get_json(silent=True) or {}
    action = body.get("action")
    if action is None:
        return jsonify({"error": "missing action"}), 400
    action = int(action)

    if env.whose_turn() != HUMAN_PID:
        return jsonify({"error": "not your turn"}), 400

    allowed = env.get_allowed_actions(HUMAN_PID)
    if action not in allowed:
        return jsonify({"error": "illegal action", "allowed": allowed}), 400

    _, _, done, info = env.step(action)
    log_step(HUMAN_PID, action, env, info)
    run_bots()

    return jsonify(state_json())


if __name__ == "__main__":
    new_game(n_players=4)
    app.run(host="127.0.0.1", port=5000, debug=False)
