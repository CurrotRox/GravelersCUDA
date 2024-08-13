My second attempt at CUDA. My first ended up a lot like Loki Scarlet's, so I didn't feel comfortable submitting it, though it was a learning experience.  
A lot of this code is the same, but I've implemented a new RNG method that I thought would remove what I thought was the bottleneck (It did)
  
![image](https://github.com/user-attachments/assets/333aa555-a61e-4dce-aacb-64aac05f6e57)  
  
Fastest I could get it to run by minimizing the load on my GPU beforehand:  
![image](https://github.com/user-attachments/assets/61d1f90b-8ad9-4717-9c2a-36a9aa30698b)  
  
I implemented XORShift32 to generate random numbers REALLY fast, and that closed the gap from about 9 seconds to barely less than a tenth of a second (on my machine.)  
Note: This can only be run on machines with an nvidia graphics card.  

Edit Stuff:  
As I've made progress, I've come back and edited my readme and my files.  
Each version of my code is in it's own folder, along with everything the compiler gave me. I included it for the sake of being transparent.  
  
Second Ver Edit:  
I was double checking everything, and came across a couple bugs.  
1. Every compilation of the .cu resulted in a different result, but it was static for the .exe
2. Every now and again, the program would spit out a 231 max, which is way more often than it should (it really shouldn't, ever)
   So I added the clock to the seed, twice. Once in the kernel (the main body of work the program does), and once in my xorshift32 implementation. This introduced extra randomness and keeps everything running fast, and randomly.
   ![image](https://github.com/user-attachments/assets/e0adc2f6-1982-4832-b0f3-014a95da6cf7)

Third Ver Edit:  
I've updated the code to now record and print the time locally instead of using the windows' Measure-Command to do it, eliminating  
1. Windows overhead
2. Any measure-command overhead.    

I've also set it up into loops on the recommendation of someone else, and while I didn't notice any performance increase, it doesn't hurt me any to have it, so I've left it in.  
The code before and after the loops is still uploaded to be examined.  
Here's some output from my code:  
![image](https://github.com/user-attachments/assets/d087fbff-3729-47f6-84de-baeee8815e3f)

Fourth Ver Edit:  
Not much has been changed this time, but I've managed to squeeze out a few more ms of performance, my current best time sitting in the 133-134ms range.  
This version has a small bug, where if I run too many threads, 231's start popping up, but with the implementation of loops, it doesn't happen.  
I've also tried moving around where the CUDA events are, at the beginning and end of as much of the program as possible, and the execution time is about the same, so I left it as is.  
![image](https://github.com/user-attachments/assets/8047863f-1e36-4c14-b62f-ea58697d3a6d)  

Fifth Ver Edit:
I've probably sunk too much time into this at this point, but for whatever reason optimizing this doesn't leave my mind. I've changed where xorshift lives, in order to cut down on the amount of callbacks and variables involved. The overall logic is the same, it's just more streamlined.  
I've also added in an additional timer, so both the total computation time, and the total kernel time. On my machine, the difference between them is ~0.5ms, but I want to be transparent about what's going on. As per usual, I've left some of my runs in a screenshot below.  
![image](https://github.com/user-attachments/assets/563c8a13-a33c-4649-9309-3ca5568b270c)

Sixth Ver Edit:  
At this point I have some kind of problem. Anyway, special shoutout to DasMonitor on Discord for wording it in a way that it clicked for me, but this new version uses bitshifting to quickly sort through the randomly generated number. Since 2 total bits make up 4 possible numbers, each 2 bits in a random number is essentially an independant random number. Using two factors of 231, I can get 231 random numbers (I don't use 100% of the random number, but that'd take too much overhead).  
![image](https://github.com/user-attachments/assets/ca9483fc-b2fb-4fc2-9291-1b595100a0b9)  

Sixth Ver v2 Edit:
I edited the amount of times I had to call the synchronous add and max, but doing that broke something, causing my found amount of ones to be higher than average. So I added a bit more randomness to my xorshift32 algorithm and it's essentially back to how it was.  
![image](https://github.com/user-attachments/assets/df44f7ca-8a5d-4716-9748-ad27f89e4d63)


