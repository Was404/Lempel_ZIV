import numpy as np
import tkinter as tk
import itertools


import tkinter as tk
import itertools



def create_matrix_view():
    global matrix_entries
    matrix_entries = []


    for i in range(int(matrix_rows_entry.get())):  # Цикл по строкам матрицы
        row = []
        for j in range(int(matrix_cols_entry.get())):  # Цикл по столбцам матрицы
            entry = tk.Entry(matrix_frame, width=5)
            entry.grid(row=i, column=j)
            row.append(entry)
        matrix_entries.append(row)
    create_matrix_button.config(state='disabled')

# Генерация таблицы истинности для заданной длины информационного слова k
def generate_truth_table(k):
    # Создание всех возможных комбинаций битовых значений длины k (0 и 1)
    return list(itertools.product([0, 1], repeat=k))

# Вычисление веса Хэмминга для заданного вектора
def hamming_weight(vector):
    # Подсчет количества ненулевых битов в векторе
    return sum(vector)

# Генерация векторов ошибок для кодового слова длины n и количеством исправляемых ошибок t
def generate_error_vectors(n, t):
    positions = range(n)  # Создание позиций в диапазоне от 0 до n-1

    # Создание всех возможных комбинаций позиций для t ошибок внутри n битов
    return [list(map(int, ''.join(['1' if i in c else '0' for i in positions]))) for c in itertools.combinations(positions, t)]
def process_matrix():
    matrix_type = matrix_type_var.get()  # Получаем тип матрицы из переменной

    if matrix_type == "Порождающая матрица":  # Проверяем, выбран ли тип порождающей матрицы

        G = [[int(entry.get()) for entry in row] for row in matrix_entries]  # Создаем матрицу G из введенных значений

        # Формируем строку текста для отображения исходной матрицы G
        original_matrix_text = "Исходная матрица:\n" + "\n".join([" ".join(map(str, row)) for row in G])

        # Транспонируем матрицу G для дальнейшей обработки
        mass = []
        for y in range(len(G[0])):
            a = []
            for i in range(len(G)):
                a.append(G[i][y])
            mass.append(a)

        num = 0
        queue = []
        # Находим порядок столбцов для приведения матрицы к стандартному виду
        for i1 in range(len(mass)):
            for i in range(len(mass)):
                count = 0
                for y in range(len(mass[i])):
                    if y == num and mass[i][y] == 1 or mass[i][y] == 0:
                        count += 1
                    else:
                        break
                if count == len(G):
                    num += 1
                    queue.append(i)

        # Переставляем столбцы для приведения к стандартному виду
        num2 = 0
        for q in range(len(queue)):
            for i in range(len(G[0])):
                if num2 == queue[q]:
                    break
                if i == queue[q]:
                    for w in range(len(G)):
                        G[w][i], G[w][num2] = G[w][num2], G[w][i]
            num2 += 1

        # Извлекаем часть матрицы P из G
        P = []
        for y in range(len(G)):
            mat_ne_null = []
            for i in range(len(G[y])):
                if i <= len(G) - 1:
                    continue
                else:
                    mat_ne_null.append(G[y][i])
            P.append(mat_ne_null)

        k = len(G)  # Длина информационного слова k
        n = len(G[0])  # Длина кодового слова n
        skorost = k / n  # Вычисляем скорость кода
        kol_kod_slov = 2 ** k  # Количество кодовых слов

        # Разделяем матрицу G на части для дальнейшей обработки
        P = [row[:len(G[0]) - len(G)] for row in G]
        P = [row[k:] for row in G]

        # Создаем единичную матрицу Ig размером len(P) x len(P)
        Ig = np.identity(len(P), dtype=int)

        # Строим систематическую матрицу G_sys (G-систематическая)
        G_sys = [list(I_row) + list(P_row) for I_row, P_row in zip(Ig, P)]

        # Транспонируем матрицу P для дальнейшего использования
        P_T = np.transpose(P)

        # Создаем единичную матрицу I размером len(P_T) x len(P_T)
        I = np.identity(len(P_T), dtype=int)

        # Строим проверочную матрицу H_sys (H-систематическая)
        H_sys = [list(P_row) + list(I_row) for P_row, I_row in zip(P_T, I)]

        # Транспонируем H_sys для дальнейшей обработки
        H_sys_transposed = np.transpose(H_sys)

        # Генерируем таблицу истинности для всех информационных слов длины k
        truth_table = generate_truth_table(k)

        # Получаем кодовые слова для каждого информационного слова
        code_words = [np.dot(info_word, G) % 2 for info_word in truth_table]

        # Создаем словарь для соответствия между кодовыми и информационными словами
        code_word_to_information_word = {tuple(code_word): info_word for code_word, info_word in
                                         zip(code_words, truth_table)}

        # Вычисляем вес Хэмминга для каждого кодового слова
        hamming_weights = [hamming_weight(code_word) for code_word in code_words]

        # Находим минимальный вес Хэмминга среди всех ненулевых кодовых слов
        non_zero_hamming_weights = [weight for weight in hamming_weights if weight != 0]
        d_min = min(non_zero_hamming_weights) if non_zero_hamming_weights else 0

        # Определяем количество обнаруживаемых ошибок и количество исправляемых ошибок
        detectable_errors = d_min - 1 if d_min > 0 else 0
        correctable_errors = (d_min - 1) // 2

        t = (d_min - 1) // 2  # Количество исправляемых ошибок

        # Генерируем векторы ошибок для исправления кодовых слов
        error_vectors = generate_error_vectors(n, t)
        error_vectors.append([0] * n)  # Добавляем вектор ошибок 000...0 (длины n)

        # Вычисляем синдромы для каждого вектора ошибок
        syndromes = [np.dot(e, H_sys_transposed) % 2 for e in error_vectors]

        # Создаем словарь для соответствия между синдромами и векторами ошибок
        syndrome_to_error_vector = {tuple(s): e for e, s in zip(error_vectors, syndromes)}

        try:
            # Преобразуем введенное слово в список целых чисел
            V = list(map(int, slovo_entry.get()))

            # Проверяем, соответствует ли длина введенного слова ожидаемой длине
            if len(V) != len(H_sys[0]):
                raise ValueError("Длина введенного слова не соответствует ожидаемой.")

            # Вычисляем синдром для введенного слова
            S = np.dot(V, np.transpose(H_sys)) % 2
            S_str = ' '.join(str(bit) for bit in S)
            syndrome_result_text = f"Синдром введенного слова V * H_sys^T: {S_str}"

            # Получаем вектор ошибок для данного синдрома
            error_vector_for_syndrome = syndrome_to_error_vector.get(tuple(S), [0] * len(S))
            error_vector_str = ' '.join(str(bit) for bit in error_vector_for_syndrome)
            error_vector_text = f"Вектор ошибок для синдрома: {error_vector_str}"

            # Вычисляем исправленное кодовое слово
            c = [(v + e) % 2 for v, e in zip(V, error_vector_for_syndrome)]
            c_str = ''.join(str(bit) for bit in c)
            corrected_code_word_text = f"Исправленное кодовое слово c: {c_str}"

            # Находим информационное слово, соответствующее исправленному кодовому слову
            information_word = code_word_to_information_word.get(tuple(c), "Код не может исправить ошибку")
            information_word_str = ''.join(str(bit) for bit in information_word)
            information_word_text = f"Ответ(i): {information_word_str}"

        except ValueError as e:
            # Обрабатываем исключение ValueError
            syndrome_result_text = str(e)

        except Exception:
            # Общая обработка остальных исключений
            syndrome_result_text = "Информационное слово не найдено"

        # Формирование текста для отображения

        P_matrix_text = "P-матрица:\n" + "\n".join([" ".join(map(str, row)) for row in P])
        P_T_matrix_text = "Транспонированная P-матрица:\n" + "\n".join([" ".join(map(str, row)) for row in P_T])
        H_sys_text = "H-матрица (H_sys):\n" + "\n".join([" ".join(map(str, row)) for row in H_sys])
        G_sys_text = "G-матрица (G_sys):\n" + "\n".join([" ".join(map(str, row)) for row in G_sys])
        #таблица кодовых слов
        table_text = "Инф. слова | Кодовые слова | Вес Хемминга\n" + "-" * 50 + "\n"
        for info_word, code_word, weight in zip(truth_table, code_words, hamming_weights):
            table_text += f"{info_word} | {code_word} | {weight}\n"

        # Обновление виджета Text для отображения результатов
        error_info_text = f"Минимальный вес Хемминга (d_min): {d_min}\n" \
                          f"Количество обнаруживаемых ошибок: {detectable_errors}\n" \
                          f"Количество исправляемых ошибок >=2t+1: {correctable_errors}"
        #h транпонированная
        H_sys_transposed_text = "Транспонированная H-матрица (H_sys^T):\n" + "\n".join(
            [" ".join(map(str, row)) for row in H_sys_transposed]) + "\n"

        # Формирование таблицы синдромов и векторов ошибок
        syndrome_table_text = "Синдромы| Векторы ошибок (e)\n" + "-" * 50 + "\n"
        for e, s in zip(error_vectors, syndromes):
            e_str = ' '.join(map(str, e))  # Преобразование вектора ошибок в строку без запятых
            s_str = ' '.join(map(str, s))  # Преобразование синдрома в строку без запятых
            syndrome_table_text += f"{s_str} | {e_str}\n"

            # Обновление виджета Text для отображения всей информации
        result_text.delete('1.0', tk.END)
        result_text.insert(tk.END,
                           f"{original_matrix_text}\nДлина кодовых слов: {n}\nДлина информационных слов: {k}\nСкорость кода: {skorost}\n{G_sys_text}\nКоличество кодовых слов: {kol_kod_slov}\n{P_matrix_text}\n{P_T_matrix_text}\n{H_sys_text}\n\n{table_text}\n{error_info_text}\n\n{H_sys_transposed_text}{syndrome_table_text}\n\n{syndrome_result_text}\n{error_vector_text}\n{corrected_code_word_text}\n{information_word_text}")


    elif matrix_type == "Проверочная матрица":
        #Код обработки проверочной матрицы H
        H = [[int(entry.get()) for entry in row] for row in matrix_entries]
        original_matrix_text = "Исходная матрица:\n" + "\n".join([" ".join(map(str, row)) for row in H])
        n = len(H[0])
        mass = []
        for y in range(len(H[0])):
            a = []
            for i in range(len(H)):
                a.append(H[i][y])
            mass.append(a)

        num = 0
        svap_stolb = []
        for i1 in range(len(mass)):
            for i in range(len(mass)):
                count = 0
                for y in range(len(mass[i])):
                    if y == num and mass[i][y] == 1:
                        count += 1
                    elif mass[i][y] == 0:
                        count += 1
                    else:
                        break
                if count == len(H):
                    num += 1
                    svap_stolb.append(i)
        num2 = len(H[0]) - len(svap_stolb)
        for q in range(len(svap_stolb)):
            for i in range(len(H[0])):
                if num2 == svap_stolb[q]:
                    break
                if i == svap_stolb[q]:
                    for w in range(len(H)):
                        H[w][i], H[w][num2] = H[w][num2], H[w][i]
            num2 += 1

        Matrix_P_t = []
        for y in range(len(H)):
            mat_ne_null = []
            for i in range(len(H[y])):
                if i >= len(H) + 1:
                    continue
                else:
                    mat_ne_null.append(H[y][i])
            Matrix_P_t.append(mat_ne_null)

        P2 = np.transpose(Matrix_P_t)
        G_sys2 = []
        for i in range(len(P2)):
            stroka2 = np.zeros(len(P2), dtype=int)  # Создаем массив нулей
            stroka2[i] = 1  # Заменяем i-й элемент на 1
            row = np.concatenate((stroka2, [int(x) for x in P2[i]]))  # Преобразуем элементы P2[i] в int и объединяем
            G_sys2.append(row)
        G_sys2 = np.array(G_sys2)

        Matrix_H_sys_t = []
        for i in range(len(H[0])):
            H_sys_T_vrem = []
            for y in range(len(H)):
                H_sys_T_vrem.append(H[y][i])
            Matrix_H_sys_t.append(H_sys_T_vrem)

        Matrix_H_sys = np.transpose(Matrix_H_sys_t)


        k2 = len(H)  # Длина информационного слова
        k = n-k2
        skorost = k / n
        kol_kod_slov = 2 ** k
        Ig = np.identity(k2, dtype=int)  # Создание единичной матрицы I размером k x k
        H_sys = [list(P_row) + list(I_row) for P_row, I_row in zip(P2, Ig)]


        # Создание таблицы истинности
        truth_table = generate_truth_table(k)
        # cловарь для преоьразования таьоицы кодоых слов
        code_words = [np.dot(info_word, G_sys2) % 2 for info_word in truth_table]
        # cловарь для преоьразования таьоицы кодоых слов
        code_word_to_information_word = {tuple(code_word): info_word for code_word, info_word in
                                         zip(code_words, truth_table)}

        hamming_weights = [hamming_weight(code_word) for code_word in code_words]

        # Нахождение минимального веса Хемминга среди ненулевых кодовых слов
        non_zero_hamming_weights = [weight for weight in hamming_weights if weight != 0]
        d_min = min(non_zero_hamming_weights) if non_zero_hamming_weights else 0
        detectable_errors = d_min - 1 if d_min > 0 else 0

        # Вычисление количества обнаруживаемых и исправляемых ошибок
        correctable_errors = (d_min - 1) // 2

        t = (d_min - 1) // 2  # Количество исправляемых ошибок
        error_vectors = generate_error_vectors(n, t)
        error_vectors = generate_error_vectors(n, t)
        error_vectors.append([0] * n)
        syndromes = [np.dot(e, Matrix_H_sys_t) % 2 for e in error_vectors]
        syndrome_to_error_vector = {tuple(s): e for e, s in zip(error_vectors, syndromes)}
        try:
            V = list(map(int, slovo_entry.get()))
            if len(V) != len(Matrix_H_sys[0]):
                raise ValueError("Длина введенного слова не соответствует ожидаемой.")

            # вычисление синдрома введенного слова
            S = np.dot(V, np.transpose(Matrix_H_sys)) % 2
            S_str = ' '.join(str(bit) for bit in S)  # Преобразование синдрома в строку
            syndrome_result_text = f"Синдром введенного слова V * H_sys^T: {S_str}"

            # Находим соответствующий вектор ошибок для полученного синдрома
            error_vector_for_syndrome = syndrome_to_error_vector.get(tuple(S), [0] * len(S))
            error_vector_str = ' '.join(
                str(bit) for bit in error_vector_for_syndrome)  # Преобразование вектора ошибок в строку
            error_vector_text = f"Вектор ошибок для синдрома: {error_vector_str}"

            # Вычисление кодового слова c по формуле c = V + e (по модулю 2)
            c = [(v + e) % 2 for v, e in zip(V, error_vector_for_syndrome)]
            c_str = ''.join(str(bit) for bit in c)
            corrected_code_word_text = f"Исправленное кодовое слово c: {c_str}"

            # Находим информационное слово, соответствующее исправленному кодовому слову
            information_word = code_word_to_information_word.get(tuple(c), "Неизвестное кодовое слово")
            information_word_str = ''.join(str(bit) for bit in information_word)
            information_word_text = f"Ответ(i): {information_word_str}"
        except ValueError as e:
            syndrome_result_text = str(e)
        except Exception:
            syndrome_result_text = "Ошибка: Неверный ввод"



        # Формирование текста для отображения
        matrix_P = "P матрица:\n" + "\n".join([" ".join(map(str, row)) for row in P2])
        matrix_G_sys = "матрица G_sys:\n" + "\n".join([" ".join(map(str, row)) for row in G_sys2])
        matrix_H_sys_text = "матрица H_sys:\n" + "\n".join([" ".join(map(str, row)) for row in Matrix_H_sys])
        # таблица кодовых слов
        table_text = "Инф. слова | Кодовые слова | Вес Хемминга\n" + "-" * 50 + "\n"
        for info_word, code_word, weight in zip(truth_table, code_words, hamming_weights):
            table_text += f"{info_word} | {code_word} | {weight}\n"

        # Обновление виджета Text для отображения результатов
        error_info_text = f"Минимальный вес Хемминга (d_min): {d_min}\n" \
                              f"Количество обнаруживаемых ошибок: {detectable_errors}\n" \
                              f"Количество исправляемых ошибок: {correctable_errors}"
        # h транпонированная
        H_sys_transposed_text = "Транспонированная H-матрица (H_sys^T):\n" + "\n".join(
            [" ".join(map(str, row)) for row in Matrix_H_sys_t]) + "\n"

        # Формирование таблицы синдромов и векторов ошибок
        syndrome_table_text = "Синдромы| Векторы ошибок (e)\n" + "-" * 50 + "\n"
        for e, s in zip(error_vectors, syndromes):
            e_str = ' '.join(map(str, e))  # Преобразование вектора ошибок в строку без запятых
            s_str = ' '.join(map(str, s))  # Преобразование синдрома в строку без запятых
            syndrome_table_text += f"{s_str} | {e_str}\n"

        # Обновление виджета Text для отображения всей информации
        result_text.delete('1.0', tk.END)
        result_text.insert(tk.END,
                            f"{original_matrix_text}\nДлина кодовых слов: {n}\nДлина информационных слов: {k}\nСкорость кода: {skorost}\nКоличество кодовых слов: {kol_kod_slov}\n{matrix_H_sys_text}\n{matrix_P}\n{matrix_G_sys}\n\n{table_text}\n{error_info_text}\n\n{H_sys_transposed_text}{syndrome_table_text}\n\n{syndrome_result_text}\n{error_vector_text}\n{corrected_code_word_text}\n{information_word_text}")

    else:
        result_text.delete('1.0', tk.END)
        result_text.insert(tk.END, "Выбран другой вид матрицы")


# Создание графического интерфейса
root = tk.Tk()


result_text = tk.Text(root, height=10, width=50)
scroll = tk.Scrollbar(root, command=result_text.yview)
result_text.configure(yscrollcommand=scroll.set)

# Пакуем виджеты Text и Scrollbar в окно
result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scroll.pack(side=tk.RIGHT, fill=tk.Y)

root.title("Графический интерфейс для матрицы")



matrix_type_var = tk.StringVar()
matrix_type_var.set("Вид матрицы")

matrix_type_label = tk.Label(root, text="Выберите вид матрицы:")
matrix_type_label.pack()
matrix_type_optionmenu = tk.OptionMenu(root, matrix_type_var, "Порождающая матрица", "Проверочная матрица")
matrix_type_optionmenu.pack()

slovo_label = tk.Label(root, text="Введите слово:")
slovo_label.pack()
slovo_entry = tk.Entry(root)
slovo_entry.pack()

matrix_size_label = tk.Label(root, text="Количество столбцов:")
matrix_size_label.pack()
matrix_cols_entry = tk.Entry(root)
matrix_cols_entry.pack()

matrix_size_label = tk.Label(root, text="Количество строк:")
matrix_size_label.pack()
matrix_rows_entry = tk.Entry(root)
matrix_rows_entry.pack()


matrix_frame = tk.Frame(root)
matrix_frame.pack()

create_matrix_button = tk.Button(root, text="Создать матрицу", command=create_matrix_view)
create_matrix_button.pack()

process_button = tk.Button(root, text="Обработать матрицу", command=process_matrix)
process_button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

matrix_entries = []

root.mainloop()