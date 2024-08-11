My second attempt at CUDA. My first ended up a lot like Loki Scarlet's, so I didn't feel comfortable, though it was a learning experience.  
A lot of this code is the same, but I've implemented a new RNG method that I thought would remove what I thought was the bottleneck (It did)

![image](https://github.com/user-attachments/assets/333aa555-a61e-4dce-aacb-64aac05f6e57)  

Fastest I could get it to run by minimizing the load on my GPU beforehand:
![image](https://github.com/user-attachments/assets/61d1f90b-8ad9-4717-9c2a-36a9aa30698b)  



I implemented XORShift32 to generate random numbers REALLY fast, and that closed the gap from about 9 seconds to barely less than a tenth of a second (on my machine.)  
Note: This can only be run on machines with an nvidia graphics card.  
  
You'll notice 2 .cu files. The Plus accidentally ran at 100 million, not 1 billion. The CudaNoCurand is the same file, but the amount is set correctly.
  
Edit:  
I was double checking everything, and came across a couple bugs.  
1. Every compilation of the .cu resulted in a different result, but it was static for the .exe
2. Every now and again, the program would spit out a 231 max, which is way more often than it should (it really shouldn't, ever)
   So I added the clock to the seed, twice. Once in the kernel (the main body of work the program does), and once in my xorshift32 implementation. This introduced extra randomness and keeps everything running fast, and randomly.
