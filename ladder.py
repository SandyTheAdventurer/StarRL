import argparse
import asyncio
import logging

import aiohttp

from sc2.client import Client
from sc2.main import _play_game
from sc2.portconfig import Portconfig
from sc2.protocol import ConnectionAlreadyClosedError

logger = logging.getLogger(__name__)


def run_ladder_game(bot):
    parser = argparse.ArgumentParser()
    parser.add_argument("--GamePort", type=int, nargs="?", help="Game port")
    parser.add_argument("--StartPort", type=int, nargs="?", help="Start port")
    parser.add_argument("--LadderServer", type=str, nargs="?", help="Ladder server")
    parser.add_argument("--OpponentId", type=str, nargs="?", help="Opponent ID")
    parser.add_argument("--RealTime", action="store_true", help="Real time flag")
    args, _unknown = parser.parse_known_args()

    host = "127.0.0.1" if args.LadderServer is None else args.LadderServer
    host_port = args.GamePort
    lan_port = args.StartPort

    bot.ai.opponent_id = args.OpponentId

    portconfig = None
    if lan_port is not None:
        ports = [lan_port + p for p in range(1, 6)]
        portconfig = Portconfig()
        portconfig.server = [ports[1], ports[2]]
        portconfig.players = [[ports[3], ports[4]]]

    game = join_ladder_game(
        host=host,
        port=host_port,
        players=[bot],
        realtime=args.RealTime,
        portconfig=portconfig,
    )
    result = asyncio.get_event_loop().run_until_complete(game)
    return result, args.OpponentId


async def join_ladder_game(
    host,
    port,
    players,
    realtime,
    portconfig,
    save_replay_as=None,
    game_time_limit=None,
):
    ws_url = f"ws://{host}:{port}/sc2api"
    session = aiohttp.ClientSession()
    ws_connection = await session.ws_connect(ws_url, timeout=120)
    client = Client(ws_connection)
    try:
        result = await _play_game(
            players[0],
            client,
            realtime,
            portconfig,
            game_time_limit,
        )
        if save_replay_as is not None:
            await client.save_replay(save_replay_as)
    except ConnectionAlreadyClosedError:
        logger.error("Connection was closed before the game ended")
        return None
    finally:
        await ws_connection.close()
        await session.close()

    return result
