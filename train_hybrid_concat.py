L = [90, 47, 8, 18, 10, 7]

# Start with the first element in a new list
S = [L[0]]

# Loop through the rest of the elements
for i in range(1, len(L)):
    flag = True
    for j in range(len(S)):
        if L[i] < S[j]:
            # Insert L[i] before S[j]
            before_j = S[:j]     # elements before index j
            new_j = [L[i]]       # the element to insert
            after_j = S[j:]      # elements from index j onward
            S = before_j + new_j + after_j
            flag = False
            break
    # If not inserted, append at the end
    if flag:
        S.append(L[i])

print(S)
