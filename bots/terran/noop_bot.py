import sc2
from sc2.bot_ai import BotAI


class NoopBot(BotAI):
    async def on_step(self, iteration: int):
        pass