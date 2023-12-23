import tkinter as tk
from tkinter import messagebox

# Constants
WIDTH = 500
HEIGHT = 500
RADIUS = 20

def lz77_encode():
    encoded_string = []
    index = 0
    input_string = entry.get("1.0", "end")
    input_string.strip()
    while index < len(input_string):
        search_buffer = input_string[max(0, index-8):index]
        search_index = max(0, index-8)

        longest_match = ''
        match_length = 0

        # Поиск наибольшего совпадения в поисковом буфере
        while search_index < index:
            match_start = index
            match = ''

            # Проверка соответствия подстроки
            while input_string[match_start] == input_string[search_index]:
                match += input_string[match_start]
                match_start += 1
                search_index += 1

                if match_start >= len(input_string) or search_index >= index:
                    break

            # Обновление наибольшего совпадения
            if len(match) > len(longest_match):
                longest_match = match
                match_length = len(longest_match)

        # Запись кодового слова в закодированную строку
        if match_length > 0:
            encoded_string.append((index - search_index, match_length, input_string[index + match_length]))
            index += match_length + 1
        else:
            encoded_string.append((0, 0, input_string[index]))
            index += 1
    messagebox.showinfo("Ответ", encoded_string)
    return encoded_string

def lz78_encode():
    input_string = entry.get("1.0", "end")
    input_string.strip()
    dictionary = {}
    encoded_string = []
    current_phrase = ""

    for char in input_string:
        current_phrase += char
        if current_phrase not in dictionary:
            dictionary[current_phrase] = len(dictionary) + 1
            encoded_string.append((dictionary[current_phrase[:-1]], char))
            current_phrase = ""

    if current_phrase in dictionary:
        encoded_string.append((dictionary[current_phrase], ""))
    messagebox.showinfo("Ответ", f"{encoded_string}")
    return encoded_string

root = tk.Tk()
#root.title = "Lempel Ziv"
root.geometry(f"{WIDTH}x{HEIGHT}")
entry = tk.Text(root, width=30, height=10, font=('Arial', 12))
entry.pack()

button77 = tk.Button(root, text="LZ77", width=20, height=5, command=lz77_encode)
button78 = tk.Button(root, text="LZ78", width=20, height=5, command=lz78_encode)
button77.pack(side="left")
button78.pack(side="right")

root.mainloop()
