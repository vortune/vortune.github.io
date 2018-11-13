#include <stdio.h>
#include <stdlib.h>
#include "mymath.h"

int main(int argc, char **argv)
{
	double a = (double)random();
	double b = (double)random();
	double c = sum(a, b);

	printf("First random is\t\t%lf\n", a);
	printf("Second random is\t\t%lf\n", b);
	printf("Summation is\t\t%lf\n", c);

	return 0;
}
