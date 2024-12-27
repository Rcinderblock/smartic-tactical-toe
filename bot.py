#!/usr/bin/env python

"""
Bot for playing tic tac toe game with multiple CallbackQueryHandlers.
"""
from copy import deepcopy
import logging
import random

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

START_MENU, CHOOSE_DIFFICULTY, CONTINUE_GAME = range(3)

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
    logger.info("User %s started the bot.", update.message.from_user.first_name)

    keyboard = [
        [InlineKeyboardButton("Играть с другом", callback_data='friend')],
        [InlineKeyboardButton("Играть с компьютером", callback_data='computer')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Выберите режим игры:", reply_markup=reply_markup)
    return START_MENU

async def start_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> int:
    """Пишется после окончания игры или до начала всего с отображением последнего состояния поля."""
    logger.info("Start_menu table was called.")
    query = update.callback_query
    await query.answer()

    # Генерируем клавиатуру с последним состоянием игры
    state = context.user_data['keyboard_state']
    keyboard = generate_keyboard(state)

    reply_markup = InlineKeyboardMarkup(keyboard)
    # Отправляем первое сообщение с результатом игры и состоянием поля
    await query.message.reply_text(f"{text}", reply_markup= reply_markup)

    # Создаем клавиатуру с кнопками выбора дальнейших действий
    action_buttons = [
        [InlineKeyboardButton("Сыграть с компьютером", callback_data='computer')],
        [InlineKeyboardButton("Сыграть с другом", callback_data='friend')],
        [InlineKeyboardButton("Завершить игру", callback_data='end')]
    ]
    reply_markup = InlineKeyboardMarkup(action_buttons)

    # Отправляем второе сообщение с предложением выбора дальнейших действий
    await query.message.reply_text("Что будем делать дальше?", reply_markup=reply_markup)

    return START_MENU



async def choose_difficulty(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Пишется после выбора режима игры с компьютером."""
    logger.info("User %s started the game with computer.", update.callback_query.from_user.first_name)

    keyboard = [
        [InlineKeyboardButton("Легкий", callback_data='easy')],
        [InlineKeyboardButton("Средний", callback_data='medium')],
        [InlineKeyboardButton("Невозможный", callback_data='impossible')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text("Выберите сложность:", reply_markup=reply_markup)
    return CHOOSE_DIFFICULTY

async def start_computer_game(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start a game with the computer."""
    logger.info("User %s started the game with computer.", update.callback_query.from_user.first_name)
    # Очистка необязательна, идет запись поверх

    difficulty = update.callback_query.data
    context.user_data['difficulty'] = difficulty
    context.user_data['keyboard_state'] = get_default_state()
    context.user_data['current_player'] = CROSS

    keyboard = generate_keyboard(context.user_data['keyboard_state'])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(f"Игра с компьютером ({difficulty}) началась! Ваш ход.", reply_markup=reply_markup)
    return CONTINUE_GAME


async def start_friend_game(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start a game with a friend."""
    logger.info("User %s started the game with a friend.", update.callback_query.from_user.first_name)
    context.user_data.clear()
    
    context.user_data['keyboard_state'] = get_default_state()
    context.user_data['current_player'] = CROSS
    
    keyboard = generate_keyboard(context.user_data['keyboard_state'])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text("Игра с другом началась! Ход крестиков.", reply_markup=reply_markup)
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


def count_minimax_score(state: list[list[str]], depth: int, alpha: int, beta: int, maximizing_player: bool) -> int:
    """
    Возвращает оценку текущего состояния игры. То есть считает score для текущего поля.
    """
    if won([cell for row in state for cell in row]):
        return 1 if maximizing_player else -1
    if all(cell != FREE_SPACE for row in state for cell in row):
        return 0
    
    if maximizing_player:
        best_score = -float('inf')
        for r in range(3):
            for c in range(3):
                if state[r][c] == FREE_SPACE:
                    state[r][c] = ZERO
                    score = count_minimax_score(state=state, depth=depth + 1, alpha=alpha, beta=beta, maximizing_player=False)
                    state[r][c] = FREE_SPACE
                    best_score = max(best_score, score)
                    alpha = max(alpha, best_score)
                    if beta <= alpha:
                        break
        return best_score
    else:
        best_score = float('inf')
        for r in range(3):
            for c in range(3):
                if state[r][c] == FREE_SPACE:
                    state[r][c] = CROSS
                    score = count_minimax_score(state=state, depth=depth + 1, alpha=alpha, beta=beta, maximizing_player=True)
                    state[r][c] = FREE_SPACE
                    best_score = min(best_score, score)
                    beta = min(beta, best_score)
                    if beta <= alpha:
                        break

        return best_score
        

def minimax_ai_move(state: list[list[str]]) -> None:
    """
    Сложный алгоритм (альфа-бета-отсечение):
        Рекурсивный алгоритм. Этот алгоритм используется в играх, в которых можно посчитать все возможные ходы.
        Одним из самых популярных представителей алгоритма является stockfish в шахматной игре.

        В данном случае мы будем использовать рекурсивный алгоритм, чтобы находить оптимальный ход для ИИ.
    """
    best_score = -float('inf')
    best_move = None
    alpha = -float('inf')
    beta = float('inf')

    for r in range(3):
        for c in range(3):
            if state[r][c] == FREE_SPACE:
                state[r][c] = ZERO  # ИИ играет за нолики
                score = count_minimax_score(state=state, depth=0, alpha=alpha, beta=beta, maximizing_player=False)
                state[r][c] = FREE_SPACE
                if score > best_score:
                    best_score = score
                    best_move = (r, c)

    if best_move:
        state[best_move[0]][best_move[1]] = ZERO

async def game(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Main processing of the game"""
    query = update.callback_query
    await query.answer()
    logger.info(f"User {query.from_user.first_name} made a move.")

    (r, c) = int(query.data[0]), int(query.data[1])
    state = context.user_data['keyboard_state']
    current_player = context.user_data.get('current_player', CROSS)
    

    # Проверяем, свободна ли клетка
    if state[r][c] == FREE_SPACE:
        state[r][c] = current_player
        
        if won([cell for row in state for cell in row]):
            winner = "Крестики" if current_player == CROSS else "Нолики"
            return await start_menu(update, context, f"{winner} выиграли!")
        
        if all(cell != FREE_SPACE for row in state for cell in row):
            return await start_menu(update, context, 'Ничья!')
        
        if 'difficulty' in context.user_data:  # Делаем ход ИИ, если игра с компьютером
            difficalty = context.user_data['difficulty']
            if difficalty == 'easy':
                random_ai_move(state)
            elif difficalty == 'medium':
                medium_ai_move(state)
            elif difficalty == 'impossible':
                minimax_ai_move(state)

            # Проверяем, выиграл ли ИИ
            if won([cell for row in state for cell in row]):
                return await start_menu(update, context, 'Компьютер выиграл!')

            # Проверяем, ничья ли
            if all(cell != FREE_SPACE for row in state for cell in row):
                return await start_menu(update, context, 'Ничья!')
        
        else:  # Если игра с другом
            context.user_data['current_player'] = ZERO if current_player == CROSS else CROSS

        # Обновляем клавиатуру и сообщение
        keyboard = generate_keyboard(state)
        reply_markup = InlineKeyboardMarkup(keyboard)
        player = 'крестиков' if context.user_data.get('current_player', CROSS) == CROSS else 'ноликов'
        await query.edit_message_text(f"Ход {player}.", reply_markup=reply_markup)

    return CONTINUE_GAME


def won(fields: list[str]) -> bool:
    """Check if crosses or zeros have won the game"""
    # Check rows
    for i in range(0, 9, 3):
        if fields[i] == fields[i+1] == fields[i+2] != FREE_SPACE:
            return True

    # Check columns
    for i in range(3):
        if fields[i] == fields[i+3] == fields[i+6] != FREE_SPACE:
            return True

    # Check diagonals
    if fields[0] == fields[4] == fields[8] != FREE_SPACE:
        return True
    if fields[2] == fields[4] == fields[6] != FREE_SPACE:
        return True

    return False


async def end(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Returns `ConversationHandler.END`, which tells the
    ConversationHandler that the conversation is over.
    """
    logger.info("Conversation ended.")

    query = update.callback_query
    await query.edit_message_text(f"Спасибо за игру!\nЧтобы запустить бота еще раз, отправьте команду /start", reply_markup=None)
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
        states = {
            START_MENU: [  # состояние "после игры", запускает меню выбора дальнейших действий
                CallbackQueryHandler(choose_difficulty, pattern='^computer$'),
                CallbackQueryHandler(start_friend_game, pattern='^friend$'),
                CallbackQueryHandler(end, pattern='^end$')
            ],
            CHOOSE_DIFFICULTY: [  # состояние "выбор сложности", второе по счету состояние, если выбрана игра с компьютером
                CallbackQueryHandler(start_computer_game, pattern='^(easy|medium|impossible)$')
            ],
            CONTINUE_GAME: [  # состояние "играть" крутится, пока не будет победы/поражения/ничьей
                CallbackQueryHandler(game, pattern='^' + f'{r}{c}' + '$')
                for r in range(3)
                for c in range(3)
            ]
        },
        fallbacks=[CommandHandler('start', start)],
    )

    # Add ConversationHandler to application that will be used for handling updates
    application.add_handler(conv_handler)

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()