import math
import numpy as np
import matplotlib.pyplot as plt
import time
from tabulate import tabulate

f_table = False # false - если справочная таблица для Терраса не посчитана, иначе true



# Вычисление последовательности для одного числа через функцию Коллатца
def collatz(n):
	if n & 1: 
		n = 3*n + 1
	else:
		n //= 2
	return n


def collatz_to_lower_n(n):
	n0 = n
	while n >= n0:
		if n & 1: 
			n = 3*n + 1
		else:
			n //= 2


# Одна итерация функции Коллатца с делением в нечетной ветке
def T(n):
    if n & 1:
        n = 3*n + 1
    n //= 2
    return n


# Вычисление количества завершающих нулей в двоичном представлении числа x
def ctz(n):
    return (n & -n).bit_length() - 1


# Проверка последовательности числа n на сходимость с помощью функций T(n) и T1(n)
# с вычислением количества "сокращаемых" шагов
def T_and_T1(n):
    n0 = n
    odd_reduction = [] # список из длин шорткатов для нечетных чисел
    even_reduction = []	# список из длин шорткатов для четных чисел
    while n >= n0:
        # T1(n)
        n += 1
        alpha = ctz(n)
        n >>= alpha
        n *= 3**alpha
        odd_reduction.append(alpha) 
        # T(n)
        n -= 1
        alpha = ctz(n)
        n >>= alpha
        even_reduction.append(alpha) 
    mean_even_reduction = sum(np.array(even_reduction)) / len(even_reduction)
    mean_odd_reduction = sum(np.array(odd_reduction)) / len(odd_reduction)
    return [n0, mean_even_reduction, mean_odd_reduction]


# Проверка последовательности числа n на сходимость с помощью функций T(n) и T1(n)
# без "лишних" операций для замера времени
def T_and_T1_for_time(n):
    n0 = n
    while n >= n0:
        # T1(n)
        n += 1
        alpha = ctz(n)
        n >>= alpha
        n *= 3**alpha
        # T(n)
        n -= 1
        alpha = ctz(n)
        n >>= alpha

def T_and_T1_for_range(n0, nk):
	info = []
	for N in range(n0, nk+1):
		info.append(T_and_T1(N))
	return np.asarray(info)

# Проверка последовательности числа n на сходимость
# с переходом только между нечетными числами для замера времени
def collatz_cto(n):
    n0 = n
    alpha = ctz(n)				
    n >>= alpha
    while n >= n0:
        # cto(n)
        while ((n & 1) != ((n >> 1) & 1)): 
            n >>= 1;
        n = n + (n >> 1) + 1


# Вычисление справочной таблицы для использования формулы Терраса
def T_b_odd(k):
	A = [] # справочная таблица
	for b in range(0, 2**k): # по всем остаткам mod2^k
		odd = 0	
		for i in range(0, k): # вычисление T_k(b) и odd(b)
			if b & 1:
				odd += 1
			b = T(b)
		A.append([b, odd])
	return A


# Проверка числа n на сходимость с выполнением k шагов за раз
# A - вспомогательная таблица
def collatz_Tk(n, A):
    n0 = n
    k = int(math.log(len(A),2))
    while n >= n0:
        b = n % 2**k
        a = n // 2**k
        cell = A[b]  #[T_k(b), odd(b)]
        n = 3**cell[1]*a + cell[0]


# Основные характеристики для последовательности числа n до 1 (стандартная функция Коллатца)
def properties_for_n(n):
    n0 = n
    even = 0        # количество четных элементов в последовательности (само n считается)
    odd = 0         # количество нечетных элементов в последовательности (1 не считается, n считается)
    glide = 0       # количество итераций до получения числа меньше исходного
    mx = n          # максимальное число в последовательности
    f_g = False
    while n > 1:
        if n >= n0 and f_g == False: 
            glide += 1
        else:
            f_g = True
        if n & 1: 
            odd += 1
        else: 
            even += 1
        n = collatz(n)
        if n > mx: mx = n
    if even == 0: compl = 0
    else: compl = odd/even  # полнота
    delay = odd + even  # количество итераций до 1
    residue = 2**even / ((3**odd) * n0)
    return [n0, glide, mx, compl, delay, residue]


# Вычисление характеристик каждого числа из диапазона [n0, nk] (стандартная функция Коллатца)
def properties_for_range(n0, nk):
	info_list = []  # список характеристик для чисел из диапазона 
	for n in range(n0, nk + 1):
		info_list.append(properties_for_n(n))
	return np.asarray(info_list)


# Функция для определения records в списке list0
def records(list0, n0):
    records = [list0[0]] # первый автоматически считается, т.к. до него не было других значений
    data_records = [n0]
    count_records = 0
    for i in range(1, len(list0)):
        if records[count_records] < list0[i]:
            records.append(list0[i])
            data_records.append(n0 + i)
            count_records += 1
    return [data_records, records]


# Построение графика records
# index - индекс характеристики в списке info
# info: [n0, glide, mx, compl, delay, residue]
def plot_records(info, index):
    n0 = int(info[0][0])     # первое число в диапазоне
    nk = int(info[-1][0])    # последнее число в диапазоне
    data = np.array([N for N in range(n0,nk+1)])
    property = info[:,index]
    property_records = records(property, n0)
    print(property_records[0])
    print(property_records[1])
    str = ''
    if index == 1: str = 'Glide Records'
    elif index == 2: str = 'Path Records'
    elif index == 3: str = 'Completeness Records'
    elif index == 4: str = 'Delay Records'
    plt.plot(data, property, 'darkblue', property_records[0], property_records[1], 'or', linewidth=0.2)
    plt.title(str, fontsize=15)
    plt.show()


# Поиск class records
def class_records(info):
	data_delay = (info[:,[0,4]]).astype(int) # в нулевом столбце n, в первом D(n)
	row_count = np.max(data_delay[:,1]) // 8 + 1 # максимально возможное значение номера строки	
	delay_records = np.zeros((row_count, 8)) # количество столбцов = 8
	for i in range(0, len(data_delay)):
		delay_int = int(data_delay[i][1]) // 8	
		delay_mod8 = int(data_delay[i][1]) % 8 
		if (delay_mod8 == 0):
			index = delay_int - 1		# число с нулевым остатком delay % 8 последнее в строке выше
			index_in_class = 7			
		else: 
			index = delay_int					# номер строки
			index_in_class = delay_mod8 - 1		# номер числа внутри строки
		if (delay_records[index][index_in_class] == 0): # нужно только минимальное число с таким значением delay
			delay_records[index][index_in_class] = data_delay[i][0]	# поэтому последующие записываться не будут
	return delay_records.astype(int)


# Построение таблицы class records
def print_class_records(delay_records):
	headers_arr = []
	headers_arr.append('N')
	for i in range(0, 9):
		headers_arr.append(f'N + {i + 1}')
	table = np.zeros((len(delay_records), 9))
	for i in range(len(delay_records)):
		table[i][0] = f'{i*8}'
		for j in range(1, 9):
			table[i][j] = delay_records[i][j-1]
	print(tabulate(table, headers = headers_arr, tablefmt="psql"))


# Количество последовательностей за секунду (график сравнения времени)
# type = 1 - стандартная функция Коллатца
# type = 2 - функции T(n) и T1(n)
# type = 3 - формула Терраса
# type = 4 - cto(n)
# проверка на диапазоне до 2**18
def Time(sieve, type):
	all_time = 0
	all_nums = 0
	time_list = []
	sec = 1
	n0 = 2
	nk = 2**20
	if type == 3:
		table = T_b_odd(10)
	for n in range(n0, nk):
		b_sieve = n % (2**20)
		if sieve[b_sieve] == 0:
			all_nums += 1
			continue
		if type == 1:
			start_time = time.time()
			collatz_to_lower_n(n)
			end_time = time.time()
		elif type == 2:
			start_time = time.time()
			T_and_T1_for_time(n)
			end_time = time.time()
		elif type == 3:
			start_time = time.time()
			collatz_Tk(n, table)
			end_time = time.time()
		else:
			start_time = time.time()
			collatz_cto(n)
			end_time = time.time()
		exec_time = end_time - start_time
		if (all_time + exec_time <= 1):
			all_time += exec_time
			all_nums += 1
		else:
			time_list.append([sec, all_nums])
			all_time = 0
			all_nums = 0
			sec += 1
	if (all_time <= 1):
		time_list.append([sec, all_nums])
	return np.asarray(time_list)


def plot_time(time_c, time_b, time_tk, time_cto):
    bw = 0.2
    max_y = max([np.max(time_c[:,1]), np.max(time_b[:,1]), np.max(time_tk[:,1]), np.max(time_cto[:,1])])
    max_x = max([np.max(time_c[:,0]), np.max(time_b[:,0]), np.max(time_tk[:,0]), np.max(time_cto[:,0])])
    plt.axis([0.5,max_x+1,0,max_y*1.1])
    plt.bar(time_tk[:,0], time_tk[:,1], bw, color='dodgerblue')
    plt.bar(time_b[:,0]+bw, time_b[:,1], bw, color='slategray')
    plt.bar(time_cto[:,0]+2*bw, time_cto[:,1], bw, color='darkblue')
    plt.bar(time_c[:,0]+3*bw, time_c[:,1], bw, color='lightblue')
    plt.xticks(time_b[:,0])
    plt.grid(axis='y', linestyle='--', alpha = 0.5)
    plt.xlabel("Секунды", fontsize=10)
    plt.ylabel("Количество последовательностей", fontsize=10)
    plt.legend(["Вычисление на k итераций вперед", "Функции T(n) и T1(n)", "Функция cto(n)", "Функция Коллатца"], prop={'size': 8})
    plt.show()


# проверка числа на сходимость за k шагов
# d - результат T_k(b)
# table_i - таблица со степенями числа i
# a*3^odd + d < a*2^k + b
# a*(3^odd - 2^k) < b - d
def is_killed_at_k(b, k):
    d = b # значение Tk(b)
    odd = d & 1
    for it in range(1,k+1):
        if d & 1:
            odd += 1
        d = T(d)
        r = 3**odd - 2**it
        l = b - d
        if r < l and r < 0: 
            return True
    return False

# вычисление сита на 2**k элементов
def collatz_sieve(k):
    sieve = np.full(2**k, 0)
    B = 2**k - 1    # максимально возможный остаток от деления n на 2^k
    for b in range(3, B + 1, 4):    # только 3mod4
        if is_killed_at_k(b,k) == False:
            sieve[b] = 1
    return sieve



str1 = """\n1 - Вычисление характеристик чисел из диапазона [n0,nk] через стандартную функцю Коллатца.
2 - Проверка чисел из диапазона [n0,nk] на сходимость с помощью метода Барины.
3 - График сравнения времени работы используемых алгоритмов.
exit - завершить работу программы.\n"""
str2 = """\n1 - Построить график glide records.
2 - Построить график path records.
3 - Построить график completeness records.
4 - Построить график delay records.
5 - Построить график residue.
6 - Вывести class records.
7 - Вернуться к основному списку команд.\n"""
str3 = """\n1 - Среднее количество сокращенных разрядов в четной ветви.
2 - Среднее количество сокращенных разрядов в нечетной ветви.
3 - Вернуться к основному списку команд.\n"""
str4 = """\n1 - Среднее количество сокращенных разрядов.
2 - Вернуться к основному списку команд.\n"""
str5 = """\n1 - Через стандартную функцию Коллатца.
2 - С использованием функций T(n) и T1(n).
3 - Через функцию с cto(n).
4 - С помощью формулы Терраса.\n"""

# здесь не работает ничего кроме cmd == 1 :)
# в графике со временем какая-то пока непонятная ошибка 
def main():
	f_sieve = False # false - если сито не вычислено, иначе true
	print(str1)
	cmd = input("Введите команду: ")
	while cmd != "exit":
		if cmd == "1" or cmd == "2":
			n0 = int(input("Первое число в диапазоне n0 = "))
			k = int(input("Последнее число в диапазоне nk = 2**k, степень k = "))
			nk = 2**k
		if cmd == "1":
			info = properties_for_range(n0, nk)
			print(str2)
			cmd = input("Введите команду: ")
			while cmd != "7":
				if int(cmd) > 0 and int(cmd) < 5:
					plot_records(info, int(cmd))
				elif cmd == "5":
					plt.plot(info[:,0], info[:,5], "darkblue", linewidth=0.3)
					plt.xlabel("n")
					plt.ylabel("Res(n)")
					plt.title("Residue")
					plt.show()
				elif cmd == "6":
					delay_records = class_records(info)
					print_class_records(delay_records)
				else:
					print("Такой команды нет.")
				print(str2)
				cmd = input("Введите команду: ")
		elif cmd == "2":
			info_T = T_and_T1_for_range(n0, nk)
			print(str3)
			cmd = input("Введите команду: ")
			while cmd != "3":
				if cmd == "1":
					MER = info_T[:,1]
					plt.plot(info_T[:,0], MER, 'b--', linewidth=0.2)
					plt.xlabel("Числа из заданного диапазона", fontsize=10)
					plt.ylabel("Количество сокращенных разрядов", fontsize=10)
					plt.show()
				elif cmd == "2":
					MOR = info_T[:,2]
					plt.plot(info_T[:,0], MOR, 'b--', linewidth=0.2)
					plt.xlabel("Числа из заданного диапазона", fontsize=10)
					plt.ylabel("Количество сокращенных разрядов", fontsize=10)
					plt.show()
				else:
					print("Такой команды нет.")
				print(str3)
				cmd = input("Введите команду: ")
		elif cmd == "3":
			print("Вычисление сита на", 2**20, "элементов", sep=' ')
			sieve = collatz_sieve(20)
			print("Количество исключенных последовательностей: ", len(sieve)-sum(sieve), end='\n\n')
			print("Диапазон от 2 до 2**20")
			time_c = Time(sieve, 1)
			print("time_c ok")
			time_b = Time(sieve,2)
			print("time_b ok")
			time_Tk = Time(sieve, 3)
			print("time_Tk ok")
			time_cto = Time(sieve,4)
			print("time_cto ok")
			plot_time(time_c, time_b, time_Tk, time_cto)
		else:
			print("Такой команды нет.")
		print(str1)
		cmd = input("Введите команду: ")
	return


main()
















