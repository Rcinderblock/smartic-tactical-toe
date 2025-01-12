import pytest
from unittest.mock import MagicMock, AsyncMock
from bot import (
    won,
    game,
    reset_game,
    end,
    get_default_state,
    CROSS,
    CONTINUE_GAME,
    ConversationHandler,
)

@pytest.fixture
def mocked_context():
    context = MagicMock()
    context.user_data = {'keyboard_state': get_default_state(), 'current_player': CROSS}
    return context

@pytest.fixture
def mocked_update():
    update = AsyncMock()
    update.callback_query = AsyncMock()
    return update

def test_won():
    # Проверка выигрышной строки
    assert won(['X', 'X', 'X', '.', '.', '.', '.', '.', '.']) == True
    # Проверка выигрышного столбца
    assert won(['X', '.', '.', 'X', '.', '.', 'X', '.', '.']) == True
    # Проверка выигрышной диагонали
    assert won(['X', '.', '.', '.', 'X', '.', '.', '.', 'X']) == True
    # Проверка отсутствия выигрышной комбинации
    assert won(['X', 'O', 'X', 'O', 'X', 'O', 'O', 'X', 'O']) == False

@pytest.mark.asyncio
async def test_game(mocked_update, mocked_context):
    mocked_update.callback_query.data = '00'  # Игрок делает ход в клетку (0, 0)

    result = await game(mocked_update, mocked_context)

    # Проверяем, что ход игрока обработан
    assert mocked_context.user_data['keyboard_state'][0][0] == CROSS
    assert result == CONTINUE_GAME

@pytest.mark.asyncio
async def test_reset_game(mocked_update, mocked_context):
    # Настраиваем начальное (законченное) состояние игры
    mocked_context.user_data['keyboard_state'] = [['X', 'O', 'X'], ['O', 'X', 'O'], ['O', 'X', 'O']]

    result = await reset_game(mocked_update, mocked_context)

    # Проверяем, что состояние игры сброшено
    assert mocked_context.user_data['keyboard_state'] == get_default_state()
    assert result == CONTINUE_GAME

@pytest.mark.asyncio
async def test_end(mocked_update, mocked_context):
    # Настраиваем начальное состояние игры
    mocked_context.user_data['keyboard_state'] = [['X', 'O', 'X'], ['O', 'X', 'O'], ['O', 'X', 'O']]

    # Вызываем функцию end с await
    result = await end(mocked_update, mocked_context)

    # Проверяем, что состояние игры сброшено и возвращено ConversationHandler.END
    assert mocked_context.user_data['keyboard_state'] == get_default_state()
    assert result == ConversationHandler.END