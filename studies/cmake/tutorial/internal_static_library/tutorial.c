#include <stdio.h>
#include <stdlib.h>
#include "mymath.h"

int main(int argc, char **argv)
{
	long int num;
	num = myrandom();
	printf("%ld\n", num);
	return 0;
}
