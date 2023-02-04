---
layout: post
title: "What is Hybrid Homomorphic Encryption and Its Applications"
summary: Combining homomorphic encryption with symmetric ciphers for a more private and efficient future
author: khoaguin
date: '2023-02-02'
category: ['private-secure-ai']
usemathjax: true
keywords: private and secure ai 
thumbnail: /assets/img/posts/cipher.jpeg
permalink: /blog/what-is-hhe
---
## TL;DR
Introduce the concept of hybrid homomorphic encryption, its use cases, a brief formulation and some [demonstration code in C++](https://github.com/khoaguin/priv-sec-ai-blog/tree/main/what-is-hhe).

_If you prefer to read on Medium and give some claps for encouragement, <a href="https://encryptedlearner.com/what-is-hybrid-homomorphic-encryption-and-its-applications-b0568b21954c" target="_blank">here</a> is the link. And please leave a <a class="github-button" href="https://github.com/khoaguin/priv-sec-ai-blog" data-color-scheme="no-preference: dark; light: dark; dark: dark;" data-icon="octicon-star" data-size="large" data-show-count="true">Star</a> if you find the article useful._

## Introduction

Privacy-preserving applications have become an important topic nowadays due to people's increasingly concerns about the privacy of their data, the prevalence of machine learning applications that require access to a vast amount of data, and new regulations such as the General Data Protection Regulation (GDPR), not to mention other ethical and financial concerns. Today, we will learn about a novel privacy-enhancing technique called Hybrid Homomorphic Encryption (HHE), which is an expansion of Homomorphic Encryption (HE).

HE is an encryption technique that allows us to perform computations on encrypted data. However, one of the problems with HE is that its ciphertexts are several orders of magnitude larger than the corresponding plaintexts. HHE aims to solve this issue by combining symmetric ciphers with HE to reduce the size of the ciphertexts and the computational resources required for the party who encrypts and sends the data (e.g. a client / data owner) at the cost of more expensive computations for the party who perform computations on the encrypted data (e.g. a server, a Cloud Service Provider, or CSP). Hence, HHE can be more suitable than HE when it comes to the client-server model of encrypted computations, especially when the client has limited computational resources and internet bandwidth, for example, phones, IoT devices, etc.

### Advantages:
- Enable computations on encrypted data hence allow privacy-preserving data analysis and applications
- Reduce the size of the ciphertext, hence reduce the required computational and bandwidth resources for the party who owns, encrypts and sends the 

### Disadvantages:
- More computationally expensive on the encrypted computation domain
- Currently still restricted to certain types of data and calculations

## Use cases
Like HE, HHE can support applications in sectors where data privacy is an important concern such as finance, healthcare, regulations, etc. Furthermore, HHE can empower applications on devices with limited computing power, memory, and network bandwidth such as embedded and IoT devices.

Example application: A home surveillance application for healthcare, where IoT devices equipped in a household taking pictures (or other signals), encrypt them and send the encrypted signals to the server. The server runs an AI algorithm on the received encrypted data and detect occasions such as people having stroke, then sends the encrypted results to the household's device that is responsible for decrypting the result and causing an alarm only when the decrypted result is positive, e.g. there are people having strokes. This way, the household can utilize the server's service while the service provider do not see any pictures or sensitive data of the household.

## Let's get into the math
Followings are some brief formulations of HE and HHE.

### Homomorphic Encryption

Before getting to HHE, we need to first understand HE. With HE, we can encrypt the data and perform operations on the encrypted data. The result when decrypted will be equivalent to the result when performing similar operations on the corresponding plaintext data. To understand HE better, I refer you to [this blog post](https://blog.openmined.org/what-is-homomorphic-encryption/) from OpenMined.

Here, let's take a look at a definition of a homomorphic public key encryption scheme which is adopted from [^1] and consists of 4 algorithms:

1. $$HE.KeyGen(1^n) → (pk, sk, evk)$$: The key generation algorithm. Here, $$n$$ is a security parameter; $$pk, sk$$ and $$evk$$ are the public key, secret key and evaluation key, respectively. We use $$pk$$ to encrypt the data, $$sk$$ to decrypt the encrypted data, and $$evk$$ to perform computations on encrypted data
2. $$HE.Enc(pk, m) → c$$: The HE encryption algorithm where $$m$$ is the plaintext data and $$c$$ is the HE encrypted data
3. $$HE.Eval(evk, f, c₁, c₂, … cᵢ) → c'$$: The evaluation algorithm  where $$f$$ is a function such as addition or multiplication, and $$c'$$ is the HE encrypted result. We should have $$HE.Dec(sk, c') = f(m₁, m₂, …, mᵢ)$$
4. $$HE.Dec(sk, c) → m$$: The HE decryption algorithm that takes $$sk$$ and the ciphertext $$c$$ to create the plaintext message $$m$$

### Hybrid Homomorphic Encryption

Instead of encrypting the data with a HE scheme which produce very large ciphertext (multiple-order expansion compared to plaintext), HHE instead encrypts them with a symmetric cipher with the expansion factor of 1 and send the symmetric ciphertexts to the server. In addition to that, the client must also send an homomorphic encrypted version of his/her symmetric key. Upon reception, the server performs the symmetric decryption algorithm homomorphically to transform the symmetric ciphertext into a homomorphic ciphertext. After that, the server can perform computations on HE encrypted data. More formally, we can define an HHE scheme (according to [^2]) that consists of 5 algorithms as following

1. $$HHE.KeyGen(1ⁿ) → (pk, sk, evk)$$: This is simply the $$HE.KeyGen$$ algorithm which produces the HE public key ($$pk$$), the secret key ($$sk$$) and the evaluation key ($$evk$$)
2. $$HHE.Enc(1ⁿ, pk, m)$$: The HHE encryption algorithm. 
First, it creates a symmetric key: $$SYM.KGen(1ⁿ) → k$$.
Then, using this symmetric key, it encrypts the plaintext message $$m$$: $$SYM.Enc(k, m) → cₛ$$. Here, $$cₛ$$ is the symmetric ciphertext that will be sent to the server. Note that $$cₛ$$ has the same size compared to $$m$$.
Furthermore, $$HHE.Enc$$ also homomorphically encrypts the symmetric key $$k$$ using $$HE.Enc(pk, k) → cₖ$$. Hence, $$cₖ$$ is the HE ciphertext of the symmetric key $$k$$, and will also be sent to the server alongside with $$cₛ$$
3. $$HHE.Decomp(evk, cₖ, cₛ) → c$$: The HHE decomposition algorithm that transforms the symmetric ciphertext $$cₛ$$ into the HE ciphertext $$c$$ by homomorphically evaluating the symmetric decryption algorithm using $$cₖ$$ and $$cₛ: HE.Eval(evk, f=SYM.Dec, cₖ, cₛ) → c$$
4. $$HHE.Eval(evk, f, c₁, . . . , cᵢ) → c'$$: The HHE evaluation algorithm that simply returns $$HE.Eval(evk, f, c₁, . . . , cᵢ)$$
5. $$HHE.Dec(sk, c)$$: The HHE decryption algorithm. It simply returns $$HE.Dec(sk, c)$$

Note that in the step 2, we have to send $$cₖ$$ and $$cₛ$$ to the server. Here, $$cₖ$$ is the HE ciphertext and can be large in size. However, we only need to send $$cₖ$$ to the server once, e.g. in a setup phase. The server can use it repeatedly in the $$HHE.Decomp$$ algorithm to turn new symmetric ciphertexts into corresponding HE ciphertexts. This is the key difference between HHE and HE: Instead of sending HE ciphertexts every time to the server which can be very bandwidth intensive, HHE sends light-weight symmetric ciphertexts instead. This trick makes HHE capable of working with resource-limited devices, as symmetric ciphers are also very light to run.


## Are you ready for some code?
Before diving in the code, let's review the protocol that we will build: We have 2 parties (a client and a server) whose actions can be summarized in 3 main steps:
1. The client creates the keys with $$HHE.KeyGen$$, encrypts the data with $$HHE.Enc$$ and send the symmetric ciphertext of his data ($$cₛ$$), the HE ciphertext of his symmetric key ($$cₖ$$), the HE keys except for the secret key $$sk$$ to the server.
2. The server performs the $$HHE.Decomp$$ algorithm and a linear transformation on the client's HE encrypted data using $$HHE.Eval$$, gets the encrypted result and sends it back to the client.
3. Upon reception, the client decrypts the result with $$HHE.Dec$$ and gets the final output in plaintext.

The [full demonstration code](https://github.com/khoaguin/priv-sec-ai-blog/tree/main/what-is-hhe) is in C++ and is built upon the the [Microsoft's SEAL](https://github.com/microsoft/SEAL) and [PASTA library](https://github.com/IAIK/hybrid-HE-framework). First, let's make 2 structs that represents the client and the server:

```cpp
struct Client
{
    // the HE keys
    seal::PublicKey he_pk;  // HE public key
    seal::SecretKey he_sk;  // HE secret key
    seal::RelinKeys he_rk;  // HE relinearization key (you don't have to care about this)
    seal::GaloisKeys he_gk; // HE galois key (you don't have to care about this)
    // client's symmetric keys
    std::vector<uint64_t> k;           // the secret symmetric keys
    std::vector<seal::Ciphertext> c_k; // the HE encrypted symmetric keys
    // client's data
    std::vector<uint64_t> m{0, 5, 255, 100, 255}; // the client's secret data
    std::vector<uint64_t> c_s;                    // the symmetric encrypted data
    seal::Ciphertext c_res;                       // the HE encrypted result received from the server
};

struct Server
{
    std::vector<int64_t> w{-1, 2, -3, 4, 5};    // dummy weights
    std::vector<int64_t> b{-5, -5, -5, -5, -5}; // dummy biases
    std::vector<seal::Ciphertext> c;            // the HE encrypted ciphertext of client's data
    seal::SecretKey he_sk;                      // the server's HE secret key
    seal::Ciphertext c_res;                     // the HE encrypted results that will be sent to the client
};

Client client;
Server server;
```

### Step 1
The client creates the SEAL context which is responsible for creating the HE keys and also other SEAL objects for encoding, encrypting and decrypting the data (BatchEncoder, Encryptor, Decryptor, Evaluator).

```cpp
std::shared_ptr<seal::SEALContext> context = sealhelper::get_seal_context();
sealhelper::print_parameters(*context);
seal::KeyGenerator keygen(*context);
keygen.create_public_key(client.he_pk);
client.he_sk = keygen.secret_key();
keygen.create_relin_keys(client.he_rk);
seal::BatchEncoder he_benc(*context);
seal::Encryptor he_enc(*context, client.he_pk);
seal::Evaluator he_eval(*context);
seal::Decryptor he_dec(*context, client.he_sk);
bool use_bsgs = false;
std::vector<int> gk_indices = pastahelper::add_gk_indices(use_bsgs, he_benc);
keygen.create_galois_keys(gk_indices, client.he_gk);
```

The client then runs the encryption algorithm ($$HHE.Enc$$) to create the symmetric key (`client.k`) and the symmetric ciphertext (`client.c_s`).

```cpp
client.k = pastahelper::get_symmetric_key();
pasta::PASTA SymmetricEncryptor(client.k, configs::plain_mod);
client.c_s = SymmetricEncryptor.encrypt(client.m);
```

If we print out the values of `client.c_s`, we will see a vector of random values such as `[30446, 62410, 62969, 38863, 43376]`, as opposed to the client's plaintext data `[0, 5, 255, 100, 255]`. The client will only send the vector of random values to the server, and never his plaintext data.

Next, the client encrypts his symmetric key (`client.k`) using HE to create `client.c_k`.

```cpp
client.c_k = pastahelper::encrypt_symmetric_key(client.k,
                                                configs::USE_BATCH,
                                                he_benc,
                                                he_enc);
```

After this, the client sends `client.c_k`, `client.c_s` and the HE keys except for the secret key to the server.

### Step 2
After receiving the `client.c_k`, the server creates his own HE secret key, the HHE object and performs the decomposition algorithm which results in `server.c` that is the HE ciphertext of the client's plaintext message `m`. Note that the client never sends his secret key `he_sk` to the server, so the server will not be able to decrypt `server.c`.

```cpp
seal::KeyGenerator csp_keygen(*context);
server.he_sk = csp_keygen.secret_key();
pasta::PASTA_SEAL HHE(context, client.he_pk, server.he_sk, client.he_rk, client.he_gk);
server.c = HHE.decomposition(client.c_s, client.c_k, configs::USE_BATCH);
```

The server then encodes his weights `w` and biases `b` and performs an element-wise vector multiplication as well as addition on his plaintext weights and biases with the HE encrypted data `server.c`.

```cpp
seal::Plaintext plain_w, plain_b;
he_benc.encode(server.w, plain_w);
he_benc.encode(server.b, plain_b);
server.c_res = sealhelper::he_mult(he_eval, server.c[0], plain_w);
client.c_res = sealhelper::he_add(he_eval, server.c_res, plain_b);
```

We can see that the final result is `client.c_res` which is the SEAL ciphertext that the client will receive.

## Step 3
Finally, the client decrypts its `c_res` using its secret key:

```cpp
std::vector<int64_t> decrypted_res = sealhelper::decrypt(client.c_res,
                                                         client.he_sk,
                                                         he_benc,
                                                         *context,
                                                         client.m.size());
```

Printing out decrypted_res, we will see that the result will be `[-5 5 -770 395 1270]`, which is correct because
`[0, 5, 255, 100, 255]
⊙ 
[-1, 2, -3, 4, 5] 
⊕ 
[-5, -5, -5, -5, -5] 
= 
[-5, 5, -770, 395, 1270]`,
where `⊙, ⊕` are the element-wise vector multiplication and addition, respectively.
The result when running the demonstration code can be seen in the below picture.

![](https://live.staticflickr.com/65535/52668112705_908350be9f_c.jpg)

## Future Directions & Conclusions

In this article, we learned about hybrid homomorphic encryption, its advantages over plain homomorphic encryption, an example use case of HHE and also walked through a very simple demonstration protocol in C++. In practice, this protocol can be extended into 3 parties which is suitable for encrypted data analysis or machine learning. You can learn more about the 3-party HHE protocol in a recently published paper [^3] at our <a href="https://research.tuni.fi/nisec/ " target="_blank">NISEC lab</a>. I hope you find this article useful and also have fun learning something new in the meantime!

## Acknowledgement

This work was funded by the <a href="https://harpocrates-project.eu/ " target="_blank">EU HARPOCRATES project</a>.


## References
[^1]: Brakerski, Zvika, and Vinod Vaikuntanathan. "Efficient fully homomorphic encryption from (standard) LWE." SIAM Journal on computing 43.2 (2014): 831–871.
[^2]: Dobraunig, Christoph, et al. "Pasta: a case for hybrid homomorphic encryption." Cryptology ePrint Archive (2021).
[^3]: Alexandros Bakas, Eugene Frimpong, Antonis Michalas. "Symmetrical Disguise: Realizing Homomorphic Encryption Services from Symmetric Primitives". EAI SECURECOMM (2022).

