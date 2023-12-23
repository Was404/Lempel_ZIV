import numpy as np

def hamming_code():
    # Шаг 1: Запрос информации о типе матрицы
    matrix_type = input("Введите тип матрицы (порождающая или проверочная): ")

    # Шаг 2: Ввод матрицы
    matrix_size = input("Введите размерность матрицы (nxk): ")
    n, k = map(int, matrix_size.split())
    matrix = []
    for _ in range(n):
        row = list(map(int, input("Введите строку матрицы через пробел: ").split()))
        matrix.append(row)

    # Шаг 3: Определение параметров n и k
    if matrix_type == "порождающая":
        n = len(matrix)
        k = len(matrix[0])
    elif matrix_type == "проверочная":
        k = len(matrix)
        n = len(matrix[0])

    # Шаг 4: Приведение матрицы к систематическому виду
    matrix = np.array(matrix)
    G = matrix[:, :k]
    P = matrix[:, k:]
    H = np.hstack((P.T, np.eye(n-k)))
    systematic_matrix = np.hstack((np.eye(k), G))

    # Шаг 5: Построение таблицы информационных и кодовых слов
    info_codewords = []
    for i in range(2**k):
        info_word = np.array(list(format(i, '0{}b'.format(k)))).astype(int)
        codeword = np.dot(info_word, G) % 2
        weight = np.count_nonzero(codeword)
        info_codewords.append((info_word.tolist(), codeword.tolist(), weight))

    # Шаг 6: Поиск веса Хэмминга и количества обнаруживающих и исправляющих ошибок
    hamming_weight = np.min([w for _, _, w in info_codewords])
    num_detect_errors = len([c for _, c, w in info_codewords if w <= hamming_weight])
    num_correct_errors = len([c for _, c, w in info_codewords if w > hamming_weight])

    # Шаг 7: Ввод кодового вектора с ошибками
    encoded_vector = input("Введите кодовый вектор с ошибками (длиной {}): ".format(n))
    encoded_vector = [int(bit) for bit in encoded_vector]

    # Шаг 8: Построение таблицы векторов ошибок и синдромов
    error_vectors_syndromes = []
    for i in range(2**n):
        error_vector = np.array(list(format(i, '0{}b'.format(n)))).astype(int)
        syndrome = np.dot(H, (encoded_vector + error_vector)) % 2
        error_vectors_syndromes.append((error_vector.tolist(), syndrome.tolist()))

    # Шаг 9: Лидерное декодирование
    decoded_word = None
    min_weight = float('inf')
    for info_word, codeword, _ in info_codewords:
        new_codeword = (codeword + syndrome) % 2
        weight = np.count_nonzero(new_codeword)
        if weight < min_weight:
            min_weight = weight
            decoded_word = codeword

    print("Результат декодирования: ", decoded_word)


if __name__ == '__main__':
    hamming_code()
