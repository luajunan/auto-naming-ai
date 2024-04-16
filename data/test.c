#include <stdio.h>
#include <string.h>

void unknown1(char *str) {
    int len = strlen(str);
    for (int i = len - 1; i >= 0; i--) {
        printf("%c", str[i]);
    }
    printf("\n");
}

int unknown2(int *arr, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum;
}

void unknown3(char *str) {
    while (*str) {
        if (*str >= 'a' && *str <= 'z') {
            printf("%c", *str - ('a' - 'A'));
        } else {
            printf("%c", *str);
        }
        str++;
    }
    printf("\n");
}
