#!/usr/bin/env python

"""
Bot for playing tic tac toe game with multiple CallbackQueryHandlers.
"""
from copy import deepcopy
import logging
from random import random

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
)
import os


# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger('httpx').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# get token using BotFather
TOKEN = os.getenv('TG_TOKEN')

CHOOSE_MODE, CHOOSE_DIFFICULTY, CONTINUE_GAME, FINISH_GAME = range(4)

FREE_SPACE = '.'
CROSS = 'X'
ZERO = 'O'


DEFAULT_STATE = [ [FREE_SPACE for _ in range(3) ] for _ in range(3) ]


def get_default_state():
    """Helper function to get default state of the game"""
    return deepcopy(DEFAULT_STATE)


def generate_keyboard(state: list[list[str]]) -> list[list[InlineKeyboardButton]]:
    """Generate tic tac toe keyboard 3x3 (telegram buttons)"""
    return [
        [
            InlineKeyboardButton(state[r][c], callback_data=f'{r}{c}')
            for r in range(3)
        ]
        for c in range(3)
    ]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Send message on `/start`."""
    logger.info("User %s started the game.", update.message.from_user.first_name)


    context.user_data['keyboard_state'] = get_default_state()
    keyboard = generate_keyboard(context.user_data['keyboard_state'])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(f'X (your) turn! Please, put X to the free place', reply_markup=reply_markup)
    return CONTINUE_GAME


def random_ai_move(state: list[list[str]]) -> None:
    """
    Простой алгоритм: делает ход в случайную свободную клетку
    """
    free_cells = [(r, c) for r in range(3) for c in range(3) if state[r][c] == FREE_SPACE]
    if free_cells:
        r, c = random.choice(free_cells)
        state[r][c] = ZERO


def medium_ai_move(state: list[list[str]]) -> None:
    """
    Средний алгоритм:
        1. Попытаться выиграть:
        Проверить, есть ли ход, который приведёт к победе ИИ. Если есть, сделать его.

        2. Блокировать игрока:
        Проверить, есть ли ход, который приведёт к победе игрока. Если есть, заблокировать его.

        3. Случайный ход:
        Если ни одна из вышеуказанных ситуаций не возникла, сделать случайный ход.
    """

    # Проверка на возможный победный ход
    for r in range(3):
        for c in range(3):
            if state[r][c] == FREE_SPACE:
                state[r][c] = ZERO
                if won([cell for row in state for cell in row]):
                    return
                state[r][c] = FREE_SPACE

    # Проверка на возможность заблокировать победу противника
    for r in range(3):
        for c in range(3):
            if state[r][c] == FREE_SPACE:
                state[r][c] = CROSS
                if won([cell for row in state for cell in row]):
                    state[r][c] = ZERO
                    return
                state[r][c] = FREE_SPACE

    # Делаем случайный ход, если ничего сверху не было найдено
    random_ai_move(state)

def minimax_ai_move(state: list[list[str]]) -> None:
    pass


async def game(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Main processing of the game"""
    # PLACE YOUR CODE HERE


def won(fields: list[str]) -> bool:
    """Check if crosses or zeros have won the game"""
    # PLACE YOUR CODE HERE


async def end(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Returns `ConversationHandler.END`, which tells the
    ConversationHandler that the conversation is over.
    """
    # reset state to default so you can play again with /start
    context.user_data['keyboard_state'] = get_default_state()
    return ConversationHandler.END


def main() -> None:
    """Run the bot"""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TOKEN).build()

    # Setup conversation handler with the states CONTINUE_GAME and FINISH_GAME
    # Use the pattern parameter to pass CallbackQueries with specific
    # data pattern to the corresponding handlers.
    # ^ means "start of line/string"
    # $ means "end of line/string"
    # So ^ABC$ will only allow 'ABC'
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            CHOOSE_MODE: [
                CallbackQueryHandler(choose_difficulty, pattern='^computer$'),
                CallbackQueryHandler(start_friend_game, pattern='^friend$')
            ],
            CHOOSE_DIFFICULTY: [
                CallbackQueryHandler(start_computer_game, pattern='^(easy|medium|hard)$')
            ],
            CONTINUE_GAME: [
                CallbackQueryHandler(game, pattern='^' + f'{r}{c}' + '$')
                for r in range(3)
                for c in range(3)
            ],
            FINISH_GAME: [
                CallbackQueryHandler(end, pattern='^' + f'{r}{c}' + '$')
                for r in range(3)
                for c in range(3)
            ],
        },
        fallbacks=[CommandHandler('start', start)],
    )

    # Add ConversationHandler to application that will be used for handling updates
    application.add_handler(conv_handler)

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()