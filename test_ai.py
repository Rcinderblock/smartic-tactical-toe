from bot import (
    random_ai_move,
    medium_ai_move,
    minimax_ai_move,
    get_default_state
)

def test_random_ai_move():
    state = get_default_state()
    random_ai_move(state)
    # Проверяем, что ИИ сделал ход (хотя бы одна клетка заполнена)
    assert state != get_default_state()

def test_medium_ai_move_block_row():
    """
    Проверка, что ИИ блокирует игрока в строке.
    """
    state = [['X', 'X', '.'], ['.', 'O', '.'], ['.', '.', '.']]
    medium_ai_move(state)
    assert state[0][2] == 'O'

def test_medium_ai_move_block_column():
    """
    Проверка, что ИИ блокирует игрока в столбце.
    """
    state = [['X', '.', '.'], ['X', 'O', '.'], ['.', '.', '.']]
    medium_ai_move(state)
    assert state[2][0] == 'O'

def test_medium_ai_move_block_diagonal():
    """
    Проверка, что ИИ блокирует игрока по диагонали.
    """
    state = [['X', '.', '.'], ['.', 'X', '.'], ['.', '.', '.']]
    medium_ai_move(state)
    assert state[2][2] == 'O'

def test_medium_ai_move_win_if_possible():
    """
    Проверка, что ИИ выигрывает, если есть возможность.
    """
    state = [['O', 'O', '.'], ['.', 'X', '.'], ['.', '.', '.']]
    medium_ai_move(state)
    assert state[0][2] == 'O'  # ИИ должен выиграть, поставив 'O' в (0, 2)


def test_medium_ai_move_no_block():
    """
    Проверка, что ИИ делает случайный ход, если нет необходимости блокировать или возможности выиграть.
    """
    state = [['X', '.', '.'], ['.', '.', '.'], ['.', '.', '.']]
    medium_ai_move(state)
    assert any(cell == 'O' for row in state for cell in row)


def test_minimax_ai_move_win():
    """
    Проверка, что ИИ выигрывает, если есть возможность.
    """
    state = [['O', 'O', '.'], ['.', 'X', '.'], ['.', '.', 'X']]
    minimax_ai_move(state)
    assert state[0][2] == 'O'

def test_minimax_ai_move_block():
    """
    Проверка, что ИИ блокирует игрока, если тот может выиграть на следующем ходу.
    """
    state = [['X', 'X', '.'], ['.', 'O', '.'], ['.', '.', '.']]
    minimax_ai_move(state)
    assert state[0][2] == 'O'  # ИИ должен заблокировать игрока, поставив 'O' в (0, 2)

def test_minimax_ai_move_optimal():
    """
    Проверка, что ИИ делает оптимальный ход, чтобы свести игру к ничьей.
    """
    state = [['.', '.', '.'], ['.', 'X', '.'], ['.', '.', '.']]
    minimax_ai_move(state)
    assert state[0][0] == 'O'  # Наиболее оптимальный ход при занятии центра первым ходом

def test_minimax_ai_move_center():
    """
    Проверка, что ИИ занимает центр, если первым ходом был занят любой из углов.
    Это наиболее оптимальный ход.
    """
    # Проверяем все возможные углы
    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    for corner in corners:
        state = [['.', '.', '.'], ['.', '.', '.'], ['.', '.', '.']]
        state[corner[0]][corner[1]] = 'X'  # Игрок занимает угол
        minimax_ai_move(state)
        # Проверяем, что ИИ занял центр
        assert state[1][1] == 'O', f"ИИ не занял центр, когда игрок занял угол {corner}"