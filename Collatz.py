import time
import math
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


# Стандартная функция Коллатца
def Collatz(N):
    if N & 1: N = 3 * N + 1
    else: N //= 2
    return N
  

# Функция Коллатца с делением на два в нечетной ветке
def Collatz_2(N):
    if N & 1: N = (3 * N + 1)//2
    else: N //=2
    return N


# Функция Коллатца с делением на два в нечетной ветке для диапазона
def CollatzForRange(N):
	N0 = N
	while N > 1:
		if N & 1: N = (3 * N + 1)//2
		else: N //= 2
	return N0


# Вычисление количества завершающих нулей в двоичном представлении числа x
def ctz(x):
    return (x & -x).bit_length() - 1


# Вычисление последовательности числа n с переходом только между нечетными числами
def Collatz_cto(N):
	odd_list = []
	alpha = ctz(N)				
	N >>= alpha
	odd_list.append(N)
	while N > 1:
		while ((N & 1) != ((N >> 1) & 1)): 
			N >>= 1;
		N = N + (N >> 1) + 1
		odd_list.append(N)
	return odd_list


# Проверка на сходимость с помощью метода Барины
def BarinaTime(N):
	N = N0
	while (N > N0):
		N += 1
		alpha = ctz(N)
		N >>= alpha
		N *= 3**alpha
		N -= 1
		beta = ctz(N)
		N >>= beta


# Расчет итераций Коллатца с помощью подхода Барины для одного числа
def CollatzBarina(N):
	N0 = N
	odd_reduction = [] # список из длин шорткатов для нечетных чисел
	even_reduction = []	# список из длин шорткатов для четных чисел
	while (N > N0):
		N += 1
		alpha = ctz(N)
		N >>= alpha
		N *= 3**alpha
		odd_reduction.append(alpha) 
		N -= 1
		beta = ctz(N)
		N >>= beta
		even_reduction.append(beta)
	mean_even_reduction = sum(np.array(even_reduction)) / len(even_reduction)
	mean_odd_reduction = sum(np.array(odd_reduction)) / len(odd_reduction)
	return [N0, mean_even_reduction, mean_odd_reduction]


# Расчет итераций Коллатца с помощью подхода Барины для диапазона чисел
def CollatzBarinaForRange(n0, nk):
	info_barina = []
	for N in range(n0, nk+1):
		info_barina.append(CollatzBarina(N))
	return np.asarray(info_barina)


# Вычисление первоначальной таблицы
# на 2^10 строк
def T_Res():
	A = []
	count = 0
	for N in range(0, 2**10):	
		odd = 0
		N0 = N
		for i in range(0, 10):
			if N & 1:
				N = (3 * N + 1)//2
				odd += 1
			else: 
				N //= 2
			if N < N0: 
				N = 0
				odd = 0
				break
		if N != 0: count += 1
		A.append([N, odd])
	print(count)
	return A


# Вычисление вперед на k итераций Коллатца 
# A - вспомогательная таблица
def CollatzTk(N, k, A):
	N0 = N
	while N > N0:
		n_l = N % 2**k
		n_h = N // 2**k
		if A[n_l][0] == 0: break
		cell = A[n_l]
		N = 3**cell[1]*n_h + cell[0]
	return N0


# Вычисление через формулу на диапазоне [n0,nk]
def CollatzTkForRange(n0, nk, k, A):
	info_Tk = []
	for N in range(n0, nk+1):
		info_Tk.append(CollatzTk(N, k, A))
	return np.asarray(info_Tk)


# Вектор четности с вычислением последовательности Коллатца и //2 в нечетной ветке
# k - длина вектора четности
def ParityVector(N, k): 
	pv = np.zeros(k)
	for i in range(0, k):
		if N & 1: 
			N = (3 * N + 1)//2
			v[i] = 1
		else: 
			N //= 2
			v[i] = 0
	return pv


# Вычисление k-го числа четности в parity vector
# с вычислением последовательности Коллатца и //2 в нечетной ветке
def ParityNumber(N, k): 
	result = 0
	deg = 0
	while (k > 0):
		if N & 1: 
			N = (3 * N + 1)//2
			result += 2**deg
		else: 
			N //= 2
		deg += 1
		k -= 1
	return result


# Перестановки для N = 2**k + N(mod2**k)
def Permutations(k):
	deg2 = 2**(k)
	count = 0
	perm = []
	for i in range(0, deg2):
		f = False
		pn = ParityNumber(2**k + i, k)
		if i != pn:
			perm.append([i, pn])
	print(perm)
	return np.asarray(perm)


# Вывод перестановок в более удобном формате
def PrintCyclicPermutations(k):
	perm = Permutations(k)
	size = perm.size // 2 - 1	# индекс последней перестановки
	cyclic_perm = np.zeros((size + 1, 2*(size + 1)), int)
	count = 0
	while size >= 0:
		f = False
		i = size - 1 
		search = perm[size][1]
		index = 0
		if perm[size][0] > 0 and perm[size][1] > 0:
			cyclic_perm[count][index] = perm[size][0]
			cyclic_perm[count][index + 1] = perm[size][1]
			index += 2
			f = True
		while i >= 0 and search > 0:
			if search == perm[i][0]:
				cyclic_perm[count][index] = perm[i][0]
				cyclic_perm[count][index + 1] = perm[i][1]
				search = perm[i][1]
				perm[i][0] = -1
				perm[i][1] = -1
				index += 2
				f = True
			i -= 1
		if f == True: count += 1
		size -= 1
	count_cyclic = 0
	for i in range(0, len(cyclic_perm)):
		if cyclic_perm[i][0] == 0: break
		print("Перестановка: ", end = '')
		for j in range(0, len(cyclic_perm[i]) - 1, 2):
			if cyclic_perm[i][j] == 0: break
			count_cyclic += 1
			print("(", cyclic_perm[i][j], ",", cyclic_perm[i][j + 1], ")", sep = '', end = ' ')
		print()
	print("Количество перестановок: ", count_cyclic)




# Основные характеристики для последовательности числа N (стандартная функция Коллатца)
def PropertiesForN(N):
    N0 = N
    even = 0        # количество четных элементов в последовательности (само N считается)
    odd = 0         # количество нечетных элементов в последовательности (1 не считается, N считается)
    glide = 0       # количество итераций до получения числа меньше исходного
    mx = N          # максимальное число в последовательности
    f_g = False
    while (N > 1):
        if (N >= N0 and f_g == False): 
            glide += 1
        else:
            f_g = True
        if N & 1: 
            odd += 1
        else: 
            even += 1
        N = Collatz(N)
        if (N > mx): mx = N
    if (even == 0): compl = 0
    else: compl = odd/even
    if (N0 == 0 or N0 == 1): gamma = 0
    else: gamma = even / math.log(N0)
    strength = 5*odd - 3*even
    level = int(math.ceil(-strength/8))
    delay = odd + even
    residue = 2**even / ((3**odd) * N0)
    return [N0, glide, mx, compl, gamma, strength, delay, level, residue]


# Вычисление характеристик каждого числа из диапазона [n0,nk] (стандартная функция Коллатца)
def PropertiesForRange(n0, nk):
	info_list = []
	i = 0
	for N in range(n0, nk + 1):
		info_list.append(PropertiesForN(N))
		i += 1
	return np.asarray(info_list)


# Функция для определения records в списке list0
def Records(list0, n0):
    records = []
    data_records = []
    records.append(list0[0]) # первый автоматически считается, т.к. до него не было других значений
    data_records.append(n0)
    count_records = 0
    for i in range(1, len(list0)):
        if records[count_records] < list0[i]:
            records.append(list0[i])
            data_records.append(n0 + i)
            count_records += 1
    return [data_records, records]


# Построение графика records
# index - индекс характеристики в списке info
def PlotRecords(info, index):
    n0 = int(info[0][0])     # первое число в диапазоне
    nk = int(info[-1][0])    # последнее число в диапазоне
    data = np.array([N for N in range(n0,nk+1)])
    property = info[:,index]
    property_records = Records(property, n0)
    print(property_records[0])
    print(property_records[1])
    str = ''
    if index == 1: str = 'Glide Records'
    elif index == 2: str = 'Path Records'
    elif index == 3: str = 'Completeness Records'
    elif index == 4: str = 'Gamma Records'
    elif index == 5: str = 'Strength Records'
    elif index == 6: str = 'Delay Records'
    plt.plot(data, property, 'darkblue', property_records[0], property_records[1], 'or', linewidth=0.2)
    plt.title(str, fontsize=15)
    plt.show()


# Поиск class records
def ClassRecords(info):
	data_delay = (info[:,[0,6]]).astype(int) # в нулевом столбце N, в первом D(N)
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


# Вывод class records
def PrintClassRecords(delay_records):
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


# Количество последовательностей за секунду
# type = 1 - стандартная функция Коллатца
# type = 2 - метод Барины
# type = 3 - формула Tk
def Time(n0, k, type):
	all_time = 0
	all_nums = 0
	time_list = []
	sec = 1
	nk = 2**k
	if type == 3:
		table = T_Res(k)
	for N in range(n0, nk + 1):
		if type == 1:
			start_time = time.time()
			CollatzForRange(N)
			end_time = time.time()
		elif type == 2:
			start_time = time.time()
			BarinaTime(N)
			end_time = time.time()
		else:
			start_time = time.time()
			CollatzTk(N, k, table)
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
	time_array = np.asarray(time_list)
	return time_array


# Количество последовательностей за секунду
def PlotTime(time_c, time_b, time_tk):
	bw = 0.3
	max_y = max([np.max(time_c[:,1]), np.max(time_b[:,1]), np.max(time_tk[:,1])])
	max_x = max([np.max(time_c[:,0]), np.max(time_b[:,0]), np.max(time_tk[:,0])])
	plt.axis([0.5,max_x+1,0,max_y + 1000])
	plt.bar(time_c[:,0], time_c[:,1], bw, color='b')
	plt.bar(time_b[:,0]+bw, time_b[:,1], bw, color='g')
	plt.bar(time_tk[:,0]+2*bw, time_tk[:,1], bw, color='r')
	plt.xticks(time_c[:,0])
	plt.grid(axis='y', linestyle='--', alpha = 0.5)
	plt.xlabel("Секунды", fontsize=10)
	plt.ylabel("Количество последовательностей", fontsize=10)
	plt.legend(["Функция Коллатца", "Метод Барины", "Вычисление на k итераций вперед"])
	plt.show()



# Количество чисел на каждом уровне
# Над столбцами указано максимальное значение полноты, встреченное на уровне
# Level = [(3*E(N) - 5*O(N)) / 8]
def LevelDistribution(n0, nk):
	info = PropertiesForRange(n0, nk)
	min_level = int(np.min(info[0:,7]))
	max_level = int(np.max(info[0:,7]))
	data = np.array([N for N in range(min_level, max_level + 1)])
	level_array = np.zeros((max_level - min_level + 1, 2)) # в 0м столбце макс. зн-е полноты, в 1м - количество
	for i in range(0, len(info)):
		index = int(info[i][7]) - min_level
		level_array[index][1] += 1
		if info[i][3] > level_array[index][0]:
			level_array[index][0] = info[i][3]
	percent = 100 / (nk - n0 + 1)
	for i in range(0, len(level_array)):
		level_array[i][1] *= percent
	plt.axis([min_level - 1,max_level + 1,0,100])
	plt.bar(data, level_array[:,1], width=0.5, facecolor='g', edgecolor='black')
	for i in range(0, len(data)):
		if level_array[i][0] > 0:
			plt.text(data[i],level_array[i][1] + min_level, '%.3f' % level_array[i][0], ha = 'center')
	plt.xticks(data)
	plt.xlabel("Уровень", fontsize=10)
	plt.ylabel("Процент от общего количества", fontsize=10)
	plt.title("Распределение чисел по уровням", fontsize=20)
	plt.grid(axis='y', linestyle='--', alpha = 0.5)
	plt.show()




str1 = """\n1 - Вычисление последовательности для заданного числа через стандартную функцию Коллатца.
2 - Вычисление последовательности, состоящей только из нечетных чисел, для заданного числа с использованием cto(x).
3 - Вычисление характеристик чисел из диапазона [n0,nk] через стандартную функцю Коллатца.
4 - Проверка чисел из диапазона [n0,nk] на сходимость с помощью метода Барины.
5 - Проверка чисел из диапазона [n0,nk] на сходимость с помощью формулы.
6 - Вычисление перестановок для значения k (N = 2**k).
7 - Количество последовательностей за секунду.
exit - завершить работу программы.\n"""
str2 = """\n1 - Построить график glide records.
2 - Построить график path records.
3 - Построить график completeness records.
4 - Построить график gamma records.
5 - Построить график strength records.
6 - Построить график delay records.
7 - Построить график residue.
8 - Вывести class records.
9 - График распределения чисел по уровням.
10 - Вернуться к основному списку команд.\n"""
str3 = """\n1 - Среднее количество сокращенных разрядов в четной ветви.
2 - Среднее количество сокращенных разрядов в нечетной ветви.
3 - Вернуться к основному списку команд.\n"""


def main():
	print(str1)
	cmd = input("Введите команду: ")
	while cmd != "exit":
		if cmd in "3457":
			n0 = int(input("n0 = "))
			k = int(input("nk = 2**k, k = "))
			nk = 2**k
		if cmd == "1":
			n = int(input("Введите натуральное число: "))
			print(n, end = " ")
			while n > 1:
				n = Collatz(n)
				print("->", n, end = " ")
			print()
		elif cmd == "2":
			n = int(input("Введите натуральное число: "))
			odd_list = Collatz_cto(n)
			for i in range(len(odd_list)):
				print(odd_list[i], end = " ")
				if odd_list[i] != 1:
					print("->", end = " ")
		elif cmd == "3":
			info = PropertiesForRange(n0, nk)
			print(str2)
			cmd = input("Введите команду: ")
			while cmd != "10":
				if int(cmd) > 0 and int(cmd) < 7:
					PlotRecords(info, int(cmd))
				elif cmd == "7":
					plt.plot(info[:,0], info[:,8], "b--", linewidth=0.3)
					plt.title("Residue")
					plt.show()
				elif cmd == "8":
					delay_records = ClassRecords(info)
					PrintClassRecords(delay_records)
				elif cmd == "9":
					LevelDistribution(n0, nk)
				else:
					print("Такой команды нет.")
				print(str2)
				cmd = input("Введите команду: ")
		elif cmd == "4":
			info_barina = CollatzBarinaForRange(n0, nk)
			print("Все числа из заданного диапазона сходятся.")
			print(str3)
			cmd = input("Введите команду: ")
			while cmd != "3":
				if cmd == "1":
					MER = info_barina[:,1]
					plt.plot(info_barina[:,0], MER, 'b--', linewidth=0.2)
					plt.xlabel("Числа из заданного диапазона", fontsize=10)
					plt.ylabel("Количество сокращенных разрядов", fontsize=10)
					plt.title("Среднее количество сокращенных разрядов в четной ветви", fontsize=15)
					plt.show()
				elif cmd == "2":
					MOR = info_barina[:,2]
					plt.plot(info_barina[:,0], MOR, 'b--', linewidth=0.2)
					plt.xlabel("Числа из заданного диапазона", fontsize=10)
					plt.ylabel("Количество сокращенных разрядов", fontsize=10)
					plt.title("Среднее количество сокращенных разрядов в нечетной ветви", fontsize=15)
					plt.show()
				else:
					print("Такой команды нет.")
				print(str3)
				cmd = input("Введите команду: ")
		elif cmd == "5":
			table = T_Res()
			CollatzTkForRange(n0, nk, k, table)
			print("Все числа из заданного диапазона сходятся.")
		elif cmd == "6":
			k = int(input("k = "))
			PrintCyclicPermutations(k)
		elif cmd == "7":
			time_c = Time(n0, k, 1)
			print("time_c ok")
			time_b = Time(n0, k, 2)
			print("time_b ok")
			time_Tk = Time(n0, k, 3)
			print("time_Tk ok")
			PlotTime(time_c, time_b, time_Tk)
		else:
			print("Такой команды нет.")
		print(str1)
		cmd = input("Введите команду: ")
	return


main()








