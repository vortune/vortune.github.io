#include <time.h>
#include "mymath.h"

long int myrandom()
{
	unsigned int seed;
	seed = (unsigned int)time(NULL);
	srandom(seed);
	long int num;
	num = random();
	return num;
}
