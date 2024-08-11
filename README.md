My second attempt at CUDA. My first ended up a lot like Loki Scarlet's, so I didn't feel comfortable, though it was a learning experience.  
A lot of this code is the same, but I've implemented a new RNG method that I thought would remove what I thought was the bottleneck (It did)

![image](https://github.com/user-attachments/assets/b5f70771-b677-4215-b66a-470237ebd627)

I implemented XORShift32 to generate random numbers REALLY fast, and that closed the gap from about 9 seconds to barely less than a tenth of a second (on my machine.)  
Note: This can only be run on machines with an nvidia graphics card.
