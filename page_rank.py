init_rank = {
    'a': 1 / 3,
    'b': 1 / 3,
    'c': 1 / 3
}

page_link_to = {
    'a': ['b', 'c'],
    'b': ['c'],
    'c': ['a']
}

pointer_by_page = {
    'a': ['c'],
    'b': ['a'],
    'c': ['a', 'b']
}

current_rank = init_rank.copy()
for i in range(10):
    temp_rank = current_rank.copy()
    for key in init_rank.keys():
        temp = 0
        for vertex_link in pointer_by_page[key]:
            temp += temp_rank[vertex_link] / len(page_link_to[vertex_link])
        current_rank[key] = temp

print(current_rank)
