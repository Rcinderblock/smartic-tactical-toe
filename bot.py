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


# Подключение логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logging.getLogger('httpx').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Получение токена
TOKEN = os.getenv('TG_TOKEN')
CHOOSE_MODE, CHOOSE_DIFFICULTY, CONTINUE_GAME, AFTER_GAME = range(4)

FREE_SPACE = '.'
CROSS = 'X'
ZERO = 'O'


DEFAULT_STATE = [ [FREE_SPACE for _ in range(3) ] for _ in range(3) ]


def get_default_state():
    """Возвращает дефолтное состояние (пустую клавиатуру в виде списка списков)"""
    return deepcopy(DEFAULT_STATE)


def generate_keyboard(state: list[list[str]], is_after_game_table: bool = False) -> list[list[InlineKeyboardButton]]:
    """Делает клавиатуру из состояния (списка списков)
    is_after_game_table - флаг, указывающий, что клавиатура генерируется после игры
    Нужен, чтобы сбрасывать игру до её завершения."""
    keyboard = [
        [
            InlineKeyboardButton(state[r][c], callback_data=f'{r}{c}')
            for r in range(3)
        ]
        for c in range(3)
    ]
    if not is_after_game_table:  # Запрещаем сбрасывать игру, если она уже завершена для избежания багов
        keyboard.append([InlineKeyboardButton("Переиграть", callback_data='reset')])  # Кнопка сброса
    return keyboard

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Пишется после команды /start. Предлагает выбор режима игры."""
    logger.info("User %s started the bot.", update.message.from_user.first_name)  # Логирование

    keyboard = [
        [InlineKeyboardButton("Играть с другом", callback_data='friend')],
        [InlineKeyboardButton("Играть с компьютером", callback_data='computer')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Привет! Я бот для игры в крестики-нолики. Ты можешь играть сам с собой, с другом или с компьютером.\nВыбери режим игры:", reply_markup=reply_markup)
    return CHOOSE_MODE


async def choose_difficulty(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Пишется после выбора режима игры с компьютером. Выбирается уровень сложности компьютера перед началом игры с ним"""
    logger.info("User %s started the game with computer.", update.callback_query.from_user.first_name)  # Логирование

    keyboard = [
        [InlineKeyboardButton("Легкий", callback_data='easy')],
        [InlineKeyboardButton("Средний", callback_data='medium')],
        [InlineKeyboardButton("Невозможный", callback_data='impossible')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text("Выберите сложность:", reply_markup=reply_markup)
    return CHOOSE_DIFFICULTY

async def start_computer_game(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало игры с компьютером с заданной сложностью."""
    # Логирование
    logger.info("User %s started the game with computer with a difficulty %s.", update.callback_query.from_user.first_name, update.callback_query.data)
    difficulty = update.callback_query.data

    context.user_data['difficulty'] = difficulty  # Установка выбранной сложности
    context.user_data['keyboard_state'] = get_default_state()  # Установка пустой клавиатуры
    context.user_data['current_player'] = CROSS  # Установка первого игрока

    keyboard = generate_keyboard(context.user_data['keyboard_state'])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(f"Игра с компьютером ({difficulty}) началась! Ваш ход.", reply_markup=reply_markup)
    return CONTINUE_GAME


async def start_friend_game(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало игры с другом (с одного tg-аккаунта)"""
    # Логирование
    logger.info("User %s started the game with a friend on the same account.", update.callback_query.from_user.first_name)
    context.user_data.clear()  # Удаляем ключ difficulty, чтобы в будущем по нему можно было определять, идет игра с компьютером или с другом.
    
    context.user_data['keyboard_state'] = get_default_state()  # Установка пустой клавиатуры
    context.user_data['current_player'] = CROSS  # Установка первого игрока
    
    keyboard = generate_keyboard(context.user_data['keyboard_state'])
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.callback_query.edit_message_text("Игра с другом началась! Ход крестиков.", reply_markup=reply_markup)
    return CONTINUE_GAME


async def reset_game(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Сбрасывает поле к начальному состоянию, универсальна для обоих режимов."""
    logger.info("User %s reset the game.")


    query = update.callback_query
    await query.answer()

    # Сбрасываем состояние игры
    context.user_data['keyboard_state'] = get_default_state()
    context.user_data['current_player'] = CROSS

    # Обновляем клавиатуру и сообщение
    keyboard = generate_keyboard(context.user_data['keyboard_state'])
    reply_markup = InlineKeyboardMarkup(keyboard)

    if 'difficulty' in context.user_data:  # Для этой строки положение "difficulty" сбрасывалось в start_friend_game
        difficulty = context.user_data['difficulty']
        await query.edit_message_text(f"Игра с компьютером ({difficulty}) началась заново! Ваш ход.", reply_markup=reply_markup)
    else:
        await query.edit_message_text("Игра с другом началась заново! Ход крестиков.", reply_markup=reply_markup)

    return CONTINUE_GAME


async def after_game(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> int:
    """Пишется после окончания игры с отображением последнего состояния поля."""
    logger.info("After_game table was called.")  # Логирование

    query = update.callback_query
    await query.answer()

    # Генерируем клавиатуру с последним состоянием игры. Это нужно, чтобы в чате оставалась история сыгранных партий.
    state = context.user_data['keyboard_state']
    keyboard = generate_keyboard(state, is_after_game_table=True)  # Генерируем без кнопки "Переиграть"

    reply_markup = InlineKeyboardMarkup(keyboard)
    # Отправляем сообщение с результатом игры и состоянием поля
    await query.edit_message_text(text, reply_markup=reply_markup)

    # Создаем клавиатуру с кнопками выбора дальнейших действий
    action_buttons = [
        [InlineKeyboardButton("Сыграть с другом", callback_data='friend')],
        [InlineKeyboardButton("Сыграть с компьютером", callback_data='computer')],
        [InlineKeyboardButton("Завершить игру", callback_data='end')]
    ]
    reply_markup = InlineKeyboardMarkup(action_buttons)

    # Отправляем второе сообщение с предложением выбора дальнейших действий
    await query.message.reply_text("Отлично сыграно!\nЧто будем делать дальше?", reply_markup=reply_markup)

    return AFTER_GAME

def random_ai_move(state: list[list[str]]) -> None:
    """
    Простой алгоритм: делает ход в случайную свободную клетку.
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
    Алгоритм минимакс для игры в крестики-нолики.
    Рекурсивная функция, которая возвращает оценку текущего состояния игры. То есть считает score для текущего поля.
    
    Args:
        state (list[list[str]]): Текущее состояние игрового поля.
        depth (int): Глубина рекурсии (сколько ходов вперед рассматривается).
        alpha (int): Лучшее значение для максимизирующего игрока (начальное значение -inf).
        beta (int): Лучшее значение для минимизирующего игрока (начальное значение +inf).
        maximizing_player (bool): Флаг, указывающий, является ли текущий игрок максимизирующим (ИИ).

    Returns:
        int: Оценка текущего состояния игры.
            1 — выигрыш максимизирующего игрока (ИИ),
            -1 — выигрыш минимизирующего игрока (игрока),
            0 — ничья.
    """

    # Проверяем, есть ли победитель на текущем поле
    if won([cell for row in state for cell in row]):
        # Если есть победитель, возвращаем 1, если выиграл минимизирующий игрок (игрок),
        # и -1, если выиграл максимизирующий игрок (ИИ).
        # Это связано с тем, что maximizing_player указывает, кто сейчас ходит,
        # а won() проверяет, выиграл ли предыдущий ход.
        return 1 if not maximizing_player else -1
    
    # Проверяем, закончилась ли игра вничью (все клетки заполнены)
    if all(cell != FREE_SPACE for row in state for cell in row):
        return 0  # Ничья
    

    # Если текущий игрок — максимизирующий (ИИ)
    if maximizing_player:
        best_score = -float('inf')
        for r in range(3):
            for c in range(3):
                if state[r][c] == FREE_SPACE:  # Перебор всех возмодных ходов
                    state[r][c] = ZERO

                    # Рекурсивно вызываем функцию для следующего хода (игрока)
                    score = count_minimax_score(state=state, depth=depth + 1, alpha=alpha, beta=beta, maximizing_player=False)
                    state[r][c] = FREE_SPACE # Отменяем ход (возвращаем поле в исходное состояние)

                    best_score = max(best_score, score) 
                    alpha = max(alpha, best_score)
                    # Если beta <= alpha, прекращаем перебор (альфа-бета отсечение). Сильно ускоряет перебор.
                    if beta <= alpha:
                        break

        return best_score  # Возвращаем лучший счет для максимизирующего игрока
    
    # Если текущий игрок — минимизирующий (игрок)
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
    Сложный алгоритм минимакс (альфа-бета-отсечение):
        Рекурсивный алгоритм. Этот алгоритм используется в играх, в которых можно посчитать все возможные ходы.
        Одним из самых популярных представителей алгоритма является stockfish в шахматной игре.

        В данном случае мы будем использовать рекурсивный алгоритм, чтобы находить оптимальный ход для ИИ.
    """
    best_score = -float('inf')
    best_move = None
    alpha = -float('inf')
    beta = float('inf')

    # Перебор всех возмодных ходов на поле
    for r in range(3):
        for c in range(3):
            if state[r][c] == FREE_SPACE:
                state[r][c] = ZERO  # ИИ делает временный ход за нолики
                score = count_minimax_score(state=state, depth=0, alpha=alpha, beta=beta, maximizing_player=False)
                state[r][c] = FREE_SPACE

                # Если оценка текущего хода лучше best_score, обновляем best_score и best_move
                if score > best_score:
                    best_score = score
                    best_move = (r, c)

    # Если лучший ход найден, делаем его
    if best_move:
        state[best_move[0]][best_move[1]] = ZERO
    else:
        # Если лучший ход не найден, выбрасываем исключение. Это означает, что ИИ не может сделать ход.
        raise Exception("No best move found.")

async def game(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обрабатывает ход игрока и, если игра с компьютером, ход ИИ.

    Args:
        update (Update): Объект, содержащий информацию о входящем сообщении или callback.
        context (ContextTypes.DEFAULT_TYPE): Контекст, содержащий данные пользователя.

    Returns:
        int: Следующее состояние игры (CONTINUE_GAME или AFTER_GAME).
    """
    query = update.callback_query
    await query.answer()
    logger.info(f"User {query.from_user.first_name} made a move.")  # Логирование



    # Получаем координаты хода из callback_data (например, "12" — строка 1, столбец 2)
    (r, c) = int(query.data[0]), int(query.data[1])

    state = context.user_data['keyboard_state']
    current_player = context.user_data.get('current_player', CROSS)
    
    # Проверяем, свободна ли выбранная клетка
    if state[r][c] == FREE_SPACE:
        state[r][c] = current_player
        
        if won([cell for row in state for cell in row]):
            winner = "Крестики" if current_player == CROSS else "Нолики"
            return await after_game(update, context, f"Вот это игра! {winner} выиграли!")
        
        if all(cell != FREE_SPACE for row in state for cell in row):
            return await after_game(update, context, 'Это было тяжело! Ничья!')
        
        if 'difficulty' in context.user_data:  # Делаем ход ИИ, если игра с компьютером
            difficulty = context.user_data['difficulty']
            if difficulty == 'easy':
                random_ai_move(state)  # Легкий уровень: случайный ход
            elif difficulty == 'medium':
                medium_ai_move(state)  # Средний уровень: блокировка игрока
            elif difficulty == 'impossible':
                minimax_ai_move(state)  # Сложный уровень: оптимальный ход

            # Проверяем, выиграл ли ИИ
            if won([cell for row in state for cell in row]):
                return await after_game(update, context, 'Этого стоило ожидать... Компьютер выиграл!')

            # Проверяем, ничья ли
            if all(cell != FREE_SPACE for row in state for cell in row):
                return await after_game(update, context, 'Ух ты! Ничья с компьютером!')
        
        # Если игра с другом, меняем текущего игрока
        else:
            context.user_data['current_player'] = ZERO if current_player == CROSS else CROSS

        # Обновляем клавиатуру и сообщение
        keyboard = generate_keyboard(state)
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Определяем, чей ход следующий
        player = 'крестиков' if context.user_data.get('current_player', CROSS) == CROSS else 'ноликов'
        await query.edit_message_text(f"Ход {player}.", reply_markup=reply_markup)

    return CONTINUE_GAME


def won(fields: list[str]) -> bool:
    """Проверяет выигранная ли игра"""

    # Проверка строк
    for i in range(0, 9, 3):
        if fields[i] == fields[i+1] == fields[i+2] != FREE_SPACE:
            return True

    # Проверка столбцов
    for i in range(3):
        if fields[i] == fields[i+3] == fields[i+6] != FREE_SPACE:
            return True

    # Проверка диагоналей   
    if fields[0] == fields[4] == fields[8] != FREE_SPACE:
        return True
    if fields[2] == fields[4] == fields[6] != FREE_SPACE:
        return True

    return False


async def end(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Завершает игру и возвращает состояние ConversationHandler.END.
    Выключает бота.

    Args:
        update (Update): Объект, содержащий информацию о входящем сообщении или callback.
        context (ContextTypes.DEFAULT_TYPE): Контекст, содержащий данные пользователя.

    Returns:
        int: ConversationHandler.END, что завершает текущий диалог.
    """
    logger.info("Conversation ended.")  # Логирование

    query = update.callback_query
    await query.edit_message_text(f"Спасибо за игру!\nЧтобы запустить бота еще раз, отправьте команду /start", reply_markup=None)
    context.user_data['keyboard_state'] = get_default_state()
    return ConversationHandler.END


def main() -> None:
    """Запуск бота"""
    # Создаем приложение для бота с использованием токена
    application = Application.builder().token(TOKEN).build()

    conv_handler = ConversationHandler(
        # Точки входа: команда /start запускает бота
        entry_points=[CommandHandler('start', start)],

        states={
            # Состояние CHOOSE_MODE: выбор режима игры (с другом или с компьютером)
            CHOOSE_MODE: [
                # Если выбран режим "компьютер", переходим к выбору сложности
                CallbackQueryHandler(choose_difficulty, pattern='^computer$'),
                # Если выбран режим "друг", начинаем игру с другом
                CallbackQueryHandler(start_friend_game, pattern='^friend$')
            ],
            # Состояние CHOOSE_DIFFICULTY: выбор сложности игры с компьютером
            CHOOSE_DIFFICULTY: [
                # Если выбрана сложность (легкая, средняя, невозможная), начинаем игру
                CallbackQueryHandler(start_computer_game, pattern='^(easy|medium|impossible)$')
            ],
            # Состояние CONTINUE_GAME: процесс игры
            CONTINUE_GAME: [
                CallbackQueryHandler(game, pattern='^' + f'{r}{c}' + '$')
                for r in range(3)
                for c in range(3)
            ] + [
                # Обработка кнопки "Переиграть"
                CallbackQueryHandler(reset_game, pattern='^reset$')
            ],
            # Состояние AFTER_GAME: игра завершена, выбор дальнейших действий
            AFTER_GAME: [
                # Если выбрана игра с компьютером, переходим к выбору сложности
                CallbackQueryHandler(choose_difficulty, pattern='^computer$'),
                # Если выбрана игра с другом, начинаем игру с другом
                CallbackQueryHandler(start_friend_game, pattern='^friend$'),
                # Если выбрано завершение игры, завершаем диалог
                CallbackQueryHandler(end, pattern='^end$')
            ]
        },
        # Fallback: если что-то пошло не так, возвращаемся к команде /start
        fallbacks=[CommandHandler('start', start)],
    )


    # Добавляем ConversationHandler в приложение
    application.add_handler(conv_handler)

    # Запускаем бота в режиме polling (постоянное ожидание обновлений)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()