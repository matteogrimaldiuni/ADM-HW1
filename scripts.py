

""""EXERCISES PROBLEM 1"""


#Introduction_Exercise_1

#script_exercise_1
if __name__ == '__main__':
    print("Hello, World!")

#script_exercise_2
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n = int(input().strip())
    
    if n % 2 != 0:
        print("Weird")
    elif n % 2 == 0 and 2 <= n <= 5:
        print("Not Weird")
    elif n % 2 == 0 and 6 <= n <= 20:
        print("Weird")
    elif n % 2 == 0 and n > 20:
        print("Not Weird")

#script_exercise_3
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    
    print(a + b)
    print(a - b)
    print(a * b)


#script_exercise_4
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    
    print(a//b)
    print(a/b)


#script_exercise_5
if __name__ == '__main__':
    n = int(input())
    
    for i in range(n):
        print(i ** 2)


#script_exercise_6
def is_leap(year):
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return True
    else:
        return False

year = int(input())
print(is_leap(year))


#script_exercise_7
if __name__ == '__main__':
    n = int(input())
    
    for i in range(1, n + 1):
        print(i, end='')

    print()


#Data types_Exercise_2


#script_exercise_1
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    
    coordinates = [[i, j, k] for i in range(x + 1) for j in range(y + 1) for k in range(z + 1) if i + j + k != n]
    print(coordinates)


#script_exercise_2
if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    arr.sort(reverse=True)
    runner_up_score = None
    for score in arr:
        if score < arr[0]:
            runner_up_score = score
            break
    print(runner_up_score)


#script_exercise_3
if __name__ == '__main__':
    
    students = []
    
    for _ in range(int(input())):
        name = input()
        score = float(input())
        
        students.append([name, score])

scores = set([student[1] for student in students])
scores.remove(min(scores))
second_lowest_score = min(scores)

second_lowest_students = [student[0] for student in students if student[1] == second_lowest_score]
second_lowest_students.sort() 

for student in second_lowest_students:
    print(student)



#script_exercise_4
if __name__ == '__main__':
    
    n = int(input())
    student_marks = {}
    
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    
    query_name = input()
    
    if query_name in student_marks:
        average_score = sum(student_marks[query_name]) / len(student_marks[query_name])
        print("{:.2f}".format(average_score))
    else:
        print("Student not found")


#script_exercise_5
if __name__ == '__main__':
    
    N = int(input())
    my_list = []
    
    for _ in range(N):
        command = input().split()
        
        if command[0] == 'insert':
            i, e = map(int, command[1:])
            my_list.insert(i, e)
        elif command[0] == 'print':
            print(my_list)
        elif command[0] == 'remove':
            e = int(command[1])
            my_list.remove(e)
        elif command[0] == 'append':
            e = int(command[1])
            my_list.append(e)
        elif command[0] == 'sort':
            my_list.sort()
        elif command[0] == 'pop':
            my_list.pop()
        elif command[0] == 'reverse':
            my_list.reverse()


#script_exercise_6
if __name__ == '__main__':
    
    n = int(input())
    integer_list = map(int, input().split())
    result = hash(tuple(integer_list))
    print(result)
    



#Strings_Excerise_3

#script_exercise_1
def swap_case(s):
    
    s2 = ""
    for x in s:
        if x.isupper() == True:
            s2 += x.lower()
        else:
            s2 += x.upper()
            
    return s2

if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)
    
    
#script_exercise_2
def split_and_join(line):
    words = line.split(" ")
    result = "-".join(words)
    return result
if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)


#script_exercise_3
def print_full_name(first, last):
    # Write your code here
    
    word = f"Hello {first} {last}! You just delved into python."
    print(word)
    
if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)
    
    
#script_exercise_4
def mutate_string(string, position, character):

    string_list = list(string)
    string_list[position] = character
    mutated_string = ''.join(string_list)
    
    return mutated_string


if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)
    
    
#script_exercise_5   
def count_substring(string, sub_string):
    
    count = 0  
    
    for i in range(0, len(string)):
        if string[i:i+len(sub_string)] == sub_string:
            count += 1
    
    return count
        

if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()
    
    count = count_substring(string, sub_string)
    print(count)
    
    
    
#script_exercise_6  
if __name__ == '__main__':
    s = input()
    
    print(any(c.isalnum() for c in s))
    print(any(c.isalpha() for c in s))
    print(any(c.isdigit() for c in s))
    print(any(c.islower() for c in s))
    print(any(c.isupper() for c in s))


#script_exercise_7
def print_hackerrank_logo(thickness):
    
    c = 'H'
    
    for i in range(thickness):
        print((c * i).rjust(thickness - 1) + c + (c * i).ljust(thickness - 1))
    for i in range(thickness + 1):
        print((c * thickness).center(thickness * 2) + (c * thickness).center(thickness * 6))
    for i in range((thickness + 1) // 2):
        print((c * thickness * 5).center(thickness * 6))
    for i in range(thickness + 1):
        print((c * thickness).center(thickness * 2) + (c * thickness).center(thickness * 6))
    for i in range(thickness):
        print(((c * (thickness - i - 1)).rjust(thickness) + c + (c * (thickness - i - 1)).ljust(thickness)).rjust(thickness * 6))


if __name__ == '__main__':
    thickness = int(input())
    print_hackerrank_logo(thickness)



#script_exercise_8
import textwrap


def wrap(string, max_width):
    wrapped_text = textwrap.wrap(string, width=max_width)
    return '\n'.join(wrapped_text)
    

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)
    
    
    
    
#script_exercise_9
def create_door_mat(rows, columns):
    
    pattern = [('.|.' * (2 * i + 1)).center(columns, '-') for i in range(rows // 2)]
    welcome = 'WELCOME'.center(columns, '-')
    door_mat = '\n'.join(pattern + [welcome] + pattern[::-1])
    return door_mat

if __name__ == '__main__':
    n, m = map(int, input().split())
    door_mat = create_door_mat(n, m)
    print(door_mat)



#script_exercise_10
def print_formatted(number):
    width = len(bin(number)) - 2
    
    for i in range(1, number + 1):
        decimal = str(i).rjust(width)
        octal = oct(i)[2:].rjust(width)
        hexadecimal = hex(i)[2:].upper().rjust(width)
        binary = bin(i)[2:].rjust(width)
        
        print(f"{decimal} {octal} {hexadecimal} {binary}")
        
        
if __name__ == '__main__':
    n = int(input())
    print_formatted(n)
    

#script_exercise_11
import string

def print_rangoli(size):
    
    alphabet = string.ascii_lowercase[:size]

    lines = []
    for i in range(size, 0, -1):
        line = '-'.join(alphabet[size-1:i-1:-1] + alphabet[i-1:size])
        lines.append(line.center(size * 4 - 3, '-'))    
    rangoli = '\n'.join(lines)
    print(rangoli)


if __name__ == '__main__':
    n = int(input())
    print_rangoli(n)
    

#script_exercise_12
def solve(s):
    words = s.split()
    capitalized_words = [word.capitalize() for word in words]
    capitalized_name = ' '.join(capitalized_words)
    return capitalized_name

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = solve(s)

    fptr.write(result + '\n')

    fptr.close()


#script_exercise_13
def minion_game(string):
    # your code goes here

    vowels = "AEIOU"
    stuart_score = 0
    kevin_score = 0

    length = len(string)

    for i in range(length):
        if string[i] in vowels:
            kevin_score += length - i
        else:
            stuart_score += length - i

    if stuart_score > kevin_score:
        print(f"Stuart {stuart_score}")
    elif stuart_score < kevin_score:
        print(f"Kevin {kevin_score}")
    else:
        print("Draw")


if __name__ == '__main__':
    s = input()
    minion_game(s)
    
    
    
#script_exercise_14
def merge_the_tools(string, k):
    num_substrings = len(string) // k
    
    for i in range(num_substrings):
        substring = string[i * k : (i + 1) * k]  
        unique_chars = []
        
        for char in substring:
            if char not in unique_chars:
                unique_chars.append(char)
        
        print(''.join(unique_chars))



if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)
    
    
    
#Sets_Excercise_4

#script_exercise_1
def average(array):
    # your code goes here
    distinct_heights = set(array)
    total_height = sum(distinct_heights)
    count = len(distinct_heights)
    avg = total_height / count
    return round(avg, 3)
if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)
    
    
#script_exercise_2
if __name__ == '__main__':
    n, m = map(int, input().split())
    arr = list(map(int, input().split()))
    like_set = set(map(int, input().split()))
    dislike_set = set(map(int, input().split()))

    happiness = 0

    for num in arr:
        if num in like_set:
            happiness += 1
        elif num in dislike_set:
            happiness -= 1

    print(happiness)


#script_exercise_3
if __name__ == '__main__':
    m = int(input())
    set_a = set(map(int, input().split()))
    n = int(input())
    set_b = set(map(int, input().split()))

    symmetric_diff = sorted(set_a.symmetric_difference(set_b))

    for num in symmetric_diff:
        print(num)


#script_exercise_4
if __name__ == '__main__':
    n = int(input())
    country_set = set()

    for _ in range(n):
        country = input()
        country_set.add(country)

    distinct_country_count = len(country_set)
    print(distinct_country_count)
    
    
#script_exercise_5
n = int(input())
s = set(map(int, input().split()))
num_commands = int(input())

for _ in range(num_commands):
    command = input().split()
    
    if command[0] == 'pop':
        s.pop()
    elif command[0] == 'remove':
        s.remove(int(command[1]))
    elif command[0] == 'discard':
        s.discard(int(command[1]))
print(sum(s))


#script_exercise_6
if __name__ == '__main__':
    n = int(input())
    english_subs = set(map(int, input().split()))
    m = int(input())
    french_subs = set(map(int, input().split()))

    total_subs = len(english_subs.union(french_subs))
    print(total_subs)


#script_exercise_7
if __name__ == '__main__':
    n = int(input())
    english_subs = set(map(int, input().split()))
    m = int(input())
    french_subs = set(map(int, input().split()))

    total_subs = len(english_subs.intersection(french_subs))
    print(total_subs)

#script_exercise_8
if __name__ == '__main__':
    n = int(input())
    english_subs = set(map(int, input().split()))
    m = int(input())
    french_subs = set(map(int, input().split()))

    english_only_subs = len(english_subs.difference(french_subs))
    print(english_only_subs)



#script_exercise_9
if __name__ == '__main__':
    n = int(input())
    english_subs = set(map(int, input().split()))
    m = int(input())
    french_subs = set(map(int, input().split()))

    total_subs = len(english_subs.symmetric_difference(french_subs))
    print(total_subs)


#script_exercise_10
if __name__ == '__main__':
    n = int(input())
    s = set(map(int, input().split()))
    num_ops = int(input())

    for _ in range(num_ops):
        operation = input().split()[0]
        other_set = set(map(int, input().split()))

        if operation == 'intersection_update':
            s.intersection_update(other_set)
        elif operation == 'update':
            s.update(other_set)
        elif operation == 'symmetric_difference_update':
            s.symmetric_difference_update(other_set)
        elif operation == 'difference_update':
            s.difference_update(other_set)

    print(sum(s))


#script_exercise_11
from collections import Counter

if __name__ == '__main__':
    k = int(input())
    room_numbers = list(map(int, input().split()))
    
    room_counts = Counter(room_numbers)
    
    for room, count in room_counts.items():
        if count == 1:
            print(room)
            break


#script_exercise_12
t = int(input())
for _ in range(t):
    n = int(input())
    set_a = set(map(int, input().split()))

    m = int(input())
    set_b = set(map(int, input().split()))

    is_subset = set_a.issubset(set_b)

    if is_subset:
        print("True")
    else:
        print("False")


#script_exercise_13
set_a = set(map(int, input().split()))
n = int(input())
is_strict_superset = True

for _ in range(n):

    set_b = set(map(int, input().split()))


    if not (set_a.issuperset(set_b) and set_a != set_b):
        is_strict_superset = False
        break
print(is_strict_superset)



#Collections_Excercise_5


#script_exercise_1
from collections import Counter

def calculate_earnings(shoe_sizes, customer_requests):
    earnings = 0
    size_counter = Counter(shoe_sizes)

    for request in customer_requests:
        size, price = request
        if size_counter[size] > 0:
            earnings += price
            size_counter[size] -= 1

    return earnings

if __name__ == "__main__":
    n = int(input())  
    shoe_sizes = list(map(int, input().split()))  #
    m = int(input())  
    customer_requests = [list(map(int, input().split())) for _ in range(m)]

    total_earnings = calculate_earnings(shoe_sizes, customer_requests)
    print(total_earnings)


#script_exercise_2
from collections import defaultdict

n, m = map(int, input().split())
group_a = defaultdict(list)
for i in range(1, n + 1):
    word = input()
    group_a[word].append(str(i))
for _ in range(m):
    word = input()
    if word in group_a:
        print(" ".join(group_a[word]))
    else:
        print("-1")


#script_exercise_3
from collections import namedtuple

n = int(input())
columns = input().split()

Student = namedtuple('Student', columns)
total_marks = 0

for _ in range(n):
    student_data = input().split()
    student = Student(*student_data)
    total_marks += int(student.MARKS)

average_marks = total_marks / n
print("{:.2f}".format(average_marks))


#script_exercise_4
from collections import OrderedDict

n = int(input())
items = OrderedDict()

for _ in range(n):
    item_data = input().split()
    item_name = ' '.join(item_data[:-1])
    item_price = int(item_data[-1])
    
    if item_name in items:
        items[item_name] += item_price
    else:
        items[item_name] = item_price

for item_name, net_price in items.items():
    print(item_name, net_price)


#script_exercise_5
n = int(input())
word_count = {}
distinct_words = []

for _ in range(n):
    word = input().strip()
    if word not in word_count:
        word_count[word] = 1
        distinct_words.append(word)
    else:
        word_count[word] += 1

print(len(distinct_words))
print(" ".join(str(word_count[word]) for word in distinct_words))



#script_exercise_6
from collections import deque

n = int(input())
d = deque()

for _ in range(n):
    operation = input().split()
    if operation[0] == "append":
        d.append(int(operation[1]))
    elif operation[0] == "appendleft":
        d.appendleft(int(operation[1]))
    elif operation[0] == "pop":
        d.pop()
    elif operation[0] == "popleft":
        d.popleft()

print(" ".join(str(x) for x in d))




#script_exercise_7
import math
import os
import random
import re
import sys

from collections import Counter


if __name__ == '__main__':
    s = input()
        
    char_counts = Counter(s)
        
    sorted_chars = sorted(char_counts.items(), key=lambda x: (-x[1], x[0]))
        
    for char, count in sorted_chars[:3]:
        print(f"{char} {count}")  



#script_exercise_8
def can_stack_cubes(test_cases):
    results = []
    
    for cubes in test_cases:
        n, blocks = cubes
        left = 0
        right = n - 1
        prev_cube = float('inf')
        valid = True
        
        while left <= right:
            if blocks[left] >= blocks[right] and blocks[left] <= prev_cube:
                prev_cube = blocks[left]
                left += 1
            elif blocks[right] >= blocks[left] and blocks[right] <= prev_cube:
                prev_cube = blocks[right]
                right -= 1
            else:
                valid = False
                break
        
        results.append("Yes" if valid else "No")
    
    return results

if __name__ == '__main__':
    T = int(input())
    test_cases = []
    
    for _ in range(T):
        n = int(input())
        blocks = list(map(int, input().split()))
        test_cases.append((n, blocks))
    
    results = can_stack_cubes(test_cases)
    
    for result in results:
        print(result)




#Date and Time_Excercise_6


#script_exercise_1
import calendar

month, day, year = map(int, input().split())

day_of_week = calendar.weekday(year, month, day)
day_name = calendar.day_name[day_of_week]
print(day_name.upper())


#script_exercise_2
import math
import os
import random
import re
import sys

from datetime import datetime, timedelta

def parse_timestamp(timestamp):
    date_obj = datetime.strptime(timestamp, '%a %d %b %Y %H:%M:%S %z')
    return date_obj

def time_delta(t1, t2):
    date1 = parse_timestamp(t1)
    date2 = parse_timestamp(t2)
    
    time_difference = abs(int((date1 - date2).total_seconds()))
    
    return str(time_difference)

if __name__ == '__main__':
    t = int(input())
    
    for _ in range(t):
        t1 = input()
        t2 = input()
        
        delta = time_delta(t1, t2)
        print(delta)




#Exceptions_Excercise_7


#script_exercise_1
if __name__ == '__main__':
    t = int(input())
    
    for _ in range(t):
        a, b = input().split()
        
        try:
            result = int(a) // int(b)
            print(result)
        except ZeroDivisionError as e:
            print("Error Code:", e)
        except ValueError as e:
            print("Error Code:", e)



#Built-ins_Excercise_8


#script_exercise_1

n, m = map(int, input().split())

marks = []

for _ in range(m):
    subject_marks = list(map(float, input().split()))
    marks.append(subject_marks)

transposed_marks = zip(*marks)
for student_marks in transposed_marks:
    average = sum(student_marks) / len(student_marks)
    print("{:.1f}".format(average))



#script_exercise_2

import math
import os
import random
import re
import sys


if __name__ == '__main__':
    first_multiple_input = input().rstrip().split()
    n = int(first_multiple_input[0])
    m = int(first_multiple_input[1])
    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input().strip())

    arr.sort(key=lambda x: x[k])


    for row in arr:
        print(" ".join(map(str, row)))



#script_exercise_3


s = input()

def custom_sort(c):
    if c.islower():
        return (1, c)
    elif c.isupper():
        return (2, c)
    elif c.isdigit() and int(c) % 2 == 1:
        return (3, c)
    else:
        return (4, c)

sorted_string = ''.join(sorted(s, key=custom_sort))

print(sorted_string)



#Python Functionals_Excercise_9

#script_exercise_1
cube = lambda x: x ** 3

def fibonacci(n):
    
    fib_list = [0, 1]
    while len(fib_list) < n:
        fib_list.append(fib_list[-1] + fib_list[-2])
    return fib_list[:n]


#Regex and Parsing challenge_Excercise_10


#script_exercise_1
import re

def is_valid_float(s):
    pattern = r'^[+-]?[0-9]*\.[0-9]+$'
    
    if re.match(pattern, s):
        return True
    else:
        return False

n = int(input())

for _ in range(n):
    test_string = input()
    if is_valid_float(test_string):
        print("True")
    else:
        print("False")

#script_exercise_2
regex_pattern = r"[.,]+"	# Do not delete 'r'.

#script_exercise_3
import re


s = input()
pattern = r'(\w)\1+'
match = re.search(pattern, s)

if match:
    print(match.group(1))
else:
    print(-1)

#script_exercise_4
import re

s = input()

pattern = r'(?<=[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM])[aeiouAEIOU]{2,}(?=[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM])'

matches = re.findall(pattern, s)

if matches:
    for match in matches:
        print(match)
else:
    print(-1)

#script_exercise_5
import re

s = input()
substring = input()


pattern = re.compile(r'(?=('+substring+'))')


matches = pattern.finditer(s)

found = False
for match in matches:
    start_index = match.start(1)
    end_index = match.end(1) - 1 
    
    print(f"({start_index}, {end_index})")
    found = True

if not found:
    print((-1, -1))

#script_exercise_6
import re


def replace_symbols(match):
    symbol = match.group(0)
    if symbol == '&&':
        return 'and'
    elif symbol == '||':
        return 'or'

n = int(input())


for _ in range(n):
    line = input()
    modified_line = re.sub(r'(?<= )(\&\&|\|\|)(?= )', replace_symbols, line)
    print(modified_line)

#script_exercise_7
regex_pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"	# Do not delete 'r'.

#script_exercise_8
import re

n = int(input())

pattern = r'^[789]\d{9}$'


for _ in range(n):
    mobile_number = input()
    
    if re.match(pattern, mobile_number):
        print("YES")
    else:
        print("NO")

#script_exercise_9

#script_exercise_10

#script_exercise_11
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(f"Start : {tag}")
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1]}")
    
    def handle_endtag(self, tag):
        print(f"End   : {tag}")
    
    def handle_startendtag(self, tag, attrs):
        print(f"Empty : {tag}")
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1]}")

n = int(input())

html_code = ""
for _ in range(n):
    html_code += input()

parser = MyHTMLParser()
parser.feed(html_code)

#script_exercise_12
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if '\n' in data:
            print(">>> Multi-line Comment")
            print(data)
        else:
            print(">>> Single-line Comment")
            print(data)

    def handle_data(self, data):
        if data.strip():
            print(">>> Data")
            print(data)

n = int(input())
html_code = ""
for _ in range(n):
    line = input()
    html_code += line + "\n"


parser = MyHTMLParser()
parser.feed(html_code)

#script_exercise_13
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1]}")

n = int(input())

html_code = ""
for _ in range(n):
    html_code += input()

parser = MyHTMLParser()
parser.feed(html_code)

#script_exercise_14
import re

def is_valid_uid(uid):
    if len(uid) != 10:
        return False

    if len(re.findall(r'[A-Z]', uid)) < 2:
        return False

    if len(re.findall(r'\d', uid)) < 3:
        return False

    if not re.match(r'^[a-zA-Z0-9]*$', uid):
        return False

    if len(set(uid)) != len(uid):
        return False

    return True

t = int(input())

for _ in range(t):
    uid = input()
    if is_valid_uid(uid):
        print('Valid')
    else:
        print('Invalid')

#script_exercise_15
import re

def is_valid_credit_card(card_number):
    pattern = r'^(4|5|6)\d{3}(-?\d{4}){3}$'

    if re.match(pattern, card_number):
        card_number = card_number.replace('-', '')
        for i in range(len(card_number) - 3):
            if card_number[i] == card_number[i+1] == card_number[i+2] == card_number[i+3]:
                return False
        return True
    else:
        return False

n = int(input())

for _ in range(n):
    card_number = input().strip()
    if is_valid_credit_card(card_number):
        print("Valid")
    else:
        print("Invalid")

#script_exercise_16

#script_exercise_17

import math
import os
import random
import re
import sys




first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])
m = int(first_multiple_input[1])

matrix = []

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)

decoded_script = ''
for j in range(m):
    for i in range(n):
        decoded_script += matrix[i][j]

decoded_script = re.sub(r'(?<=[A-Za-z0-9])[^A-Za-z0-9]+(?=[A-Za-z0-9])', ' ', decoded_script)

print(decoded_script)



#XML_Excercise_10

#script_exercise_1
def get_attr_number(node):
    score = len(node.attrib)

    for child in node:
        score += get_attr_number(child)

    return score

#script_exercise_2
maxdepth = 0

def depth(elem, level):
    global maxdepth
    # Update maxdepth if the current level is greater
    if level > maxdepth:
        maxdepth = level
    # Recursively calculate depth for child elements
    for child in elem:
        depth(child, level + 1)



#Closures and Decoration_Excercise_11

#script_exercise_1
def person_lister(f):
    def inner(people):
        sorted_people = sorted(people, key=lambda x: int(x[2]))
        
        return [f(person) for person in sorted_people]
    
    return inner

#script_exercise_2

def wrapper(f):
    def fun(l):
        # complete the function
        formatted_numbers = []
        for number in l:
            if len(number) == 10:
                formatted_numbers.append("+91 " + number[:5] + " " + number[5:])
            elif len(number) == 11 and number[0] == '0':
                formatted_numbers.append("+91 " + number[1:6] + " " + number[6:])
            elif len(number) == 12 and number[:2] == "91":
                formatted_numbers.append("+91 " + number[2:7] + " " + number[7:])
            else:
                formatted_numbers.append(number)
        
        f(formatted_numbers)
    return fun


#Numpy_Excercise_12

#script_exercise_1
import numpy as np

def arrays(arr):
    # complete this function
    # use numpy.array
    
    arr = np.array(arr, float)
    arr = arr[::-1]
    
    return arr


#script_exercise_2

import numpy as np

input_data = input().strip().split()
num_array = np.array(input_data, int).reshape(3, 3)
print(num_array)

#script_exercise_3

import numpy as np


rows, columns = map(int, input().split())
matrix = [list(map(int, input().split())) for _ in range(rows)]
array = np.array(matrix)
print(np.transpose(array))
print(array.flatten())

#script_exercise_4

import numpy as np


X, Y, Z = map(int, input().split())
arrays = []

for _ in range(X + Y):
    array = list(map(int, input().split()))
    arrays.append(array)
array_1 = np.array(arrays[:X])
array_2 = np.array(arrays[X:])
result = np.concatenate((array_1, array_2), axis=0)
print(result)

#script_exercise_5


import numpy as np

shape = tuple(map(int, input().split()))
zeros_array = np.zeros(shape, dtype=int)
ones_array = np.ones(shape, dtype=int)
print(zeros_array)
print(ones_array)

#script_exercise_6

import numpy as np


n, m = map(int, input().split())

np.set_printoptions(sign=' ')
identity_array = np.eye(n, m)

print(identity_array)

#script_exercise_7

import numpy as np


n, m = map(int, input().split())

array_a = np.array([input().split() for _ in range(n)], int)
array_b = np.array([input().split() for _ in range(n)], int)

addition_result = np.add(array_a, array_b)
subtraction_result = np.subtract(array_a, array_b)
multiplication_result = np.multiply(array_a, array_b)
division_result = np.floor_divide(array_a, array_b)
modulus_result = np.mod(array_a, array_b)
power_result = np.power(array_a, array_b)

print(addition_result)
print(subtraction_result)
print(multiplication_result)
print(division_result)
print(modulus_result)
print(power_result)

#script_exercise_8

import numpy as np


arr = np.array(input().split(), float)

np.set_printoptions(sign=' ')
print(np.floor(arr))
print(np.ceil(arr))
print(np.rint(arr))

#script_exercise_9

import numpy as np


n, m = map(int, input().split())


arr = np.array([input().split() for _ in range(n)], int)


result = np.prod(np.sum(arr, axis=0))

print(result)

#script_exercise_10

import numpy as np


n, m = map(int, input().split())

arr = np.array([input().split() for _ in range(n)], int)
result = np.max(np.min(arr, axis=1))

print(result)

#script_exercise_11

import numpy as np

n, m = map(int, input().split())

arr = np.array([input().split() for _ in range(n)], int)
mean_arr = np.mean(arr, axis=1)
var_arr = np.var(arr, axis=0)
std_arr = np.std(arr)

print(mean_arr)
print(var_arr)
print(std_arr)

#script_exercise_12

import numpy as np

n = int(input())
A = []
B = []


for _ in range(n):
    A.append(list(map(int, input().split())))


for _ in range(n):
    B.append(list(map(int, input().split())))


arr_A = np.array(A)
arr_B = np.array(B)


result = np.dot(arr_A, arr_B)

print(result)

#script_exercise_13

import numpy as np


A = np.array(list(map(int, input().split())))
B = np.array(list(map(int, input().split())))


inner_product = np.inner(A, B)
outer_product = np.outer(A, B)

print(inner_product)
print(outer_product)

#script_exercise_14

import numpy as np

coefficients = list(map(float, input().split()))
x = float(input())

result = np.polyval(coefficients, x)
print(result)

#script_exercise_15

import numpy as np

n = int(input())

matrix = [list(map(float, input().split())) for _ in range(n)]
determinant = np.linalg.det(matrix)

print(round(determinant, 2))










"""EXCERCISES PROBLEM 2"""


#Birthday Cakes Candles_Excercise_1

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
    
    max_height = max(candles)
    count = candles.count(max_height)
    return count

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

#Number Line Jumps_Excercise_2

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#

def kangaroo(x1, v1, x2, v2):
    # Write your code here

    if x1 == x2 and v1 == v2:
        return "YES"
    elif x1 != x2 and v1 == v2:
        return "NO"
    else:
        if (x2 - x1) % (v1 - v2) == 0 and (x2 - x1) / (v1 - v2) >= 0:
            return "YES"
        else:
            return "NO"

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()


#Viral Advertising_Excercise_3

import math
import os
import random
import re
import sys

#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#

def viralAdvertising(n):
    
    shared = 5 
    cumulative_likes = 0 
    
    for day in range(1, n+1):
        liked_today = shared // 2
        cumulative_likes += liked_today
        shared = liked_today * 3
    
    return cumulative_likes

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()


#Recursive Digit Sum_Excercise_4

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k
#

def superDigit(n, k):
    # Write your code here
    def calculate_super_digit(s):
        if len(s) == 1:
            return int(s)
        else:
            digit_sum = sum(int(digit) for digit in s)
            return calculate_super_digit(str(digit_sum))

    initial_super_digit = calculate_super_digit(n)
    repeated_super_digit = calculate_super_digit(str(initial_super_digit * k))

    return repeated_super_digit
    
    

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

#Insertion Sort - Part 1_Excercise_5

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort1(n, arr):
    # Write your code here

    value_to_insert = arr[-1]
    index = n - 2

    while index >= 0 and arr[index] > value_to_insert:
        arr[index + 1] = arr[index]
        print(" ".join(map(str, arr)))
        index -= 1
    
    arr[index + 1] = value_to_insert
    print(" ".join(map(str, arr)))

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)


#Insertion Sort - Part 2_Excercise_6
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort2' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort2(n, arr):
    # Write your code here
    
    for i in range(1, n):
        current_element = arr[i]
        j = i - 1
        
        while j >= 0 and current_element < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = current_element
        
        print(" ".join(map(str, arr)))

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)
