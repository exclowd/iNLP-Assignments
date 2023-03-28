# CBOW and SVD Word Embeddings

## SVD

### How to run
- with vocab
```
python svd.py ./data --vocab=./vocab.txt
```
- without vocab
```
python svd.py ./data
```

### Experiments
1. with window size = 1
```
Neighbors of 'titanic':
616. baseball
1879. halloween
279. tv
1522. hbo
236. war
1269. awful
360. white
886. fiction
1546. noir
```
2. with window size 5, and number of sentences 1000

Neighbors of 'mother':
```
841. mother
363. father
30. he
275. parents
544. alexander
915. drum
673. baby
1166. reunion
890. musical
283. three
85. him
28. his
529. house
47. has
382. especially
177. her
292. down
```
Neighbors of 'father':
```
363. father
174. brothers
298. jacob
30. he
28. his
275. parents
841. mother
85. him
1166. reunion
895. talent
161. people
115. god
1244. identity
544. alexander
382. especially
123. life
```
- with window size 5, and number of sentences 10000
Neighbors of 'mother' and 'father':
Neighbors of 'mother':
```
317. mother
318. mary
1200. brothers
1580. mission
433. father
1437. followers
28. his
393. body
2035. apostles
2201. beloved
828. loving
1378. career
723. teachings
1147. mom
417. playing
```
Neighbors of 'father':
```
434. father
1200. brothers
510. upon
555. name
2278. hat
1132. towards
789. judas
457. audience
721. holy
203. cross
317. mother
2035. apostles
329. words
24. he
1110. crowd
1337. divine
393. body
256. message
736. spirit
```

For 100000 iters
```
100001it [00:54, 1849.62it/s]
Vocab size: 35273
100001it [07:13, 230.88it/s]
(35273, 35273)
Neighbors of 'mother':
400. mother
356. father
986. sister
6227. roommate
4069. ex
616. brother
1480. marriage
4060. conscience
1629. personality
695. voice
5023. vera
1308. lover
347. son
1640. fellow
2600. uncle
4413. cousin
2310. boss
4534. aunt
1315. self
938. mary
2629. kelly
2842. mathilda
1612. helen
1422. hair
5124. wing
1869. lisa
5326. mistress
1870. identity
2638. anna
9214. pregnancy
Neighbors of 'father':
356. father
616. brother
2600. uncle
400. mother
2310. boss
5326. mistress
1640. fellow
5124. wing
3923. persona
2123. partner
2336. neighbor
5180. ego
5141. mentor
346. wife
4238. nose
5460. claudia
4464. visits
2132. losing
2794. buddy
6724. profession
1870. identity
1629. personality
1941. skills
695. voice
6889. guardian
1480. marriage
1406. girlfriend
1440. danny
6260. sanity
347. son
Neighbors of 'titanic':
1624. titanic
1630. apollo
830. third
1312. holocaust
227. last
159. original
771. actual
553. title
4632. saga
3581. royal
4040. outcome
262. during
2369. regarding
3169. fourth
2386. revolution
2355. trilogy
898. previous
448. final
1706. matrix
2100. text
1498. west
3161. heat
2141. latter
4524. messenger
7816. scarlet
4668. lion
2358. bbc
1707. industry
3584. concerning
1876. fifth
```

## CBOW
How to run
- to train and save the model in `model.pt`
```
python cbow.py ./data
```
- to test and compute accuracy scores
```
python cbow.py ./data --testmodel=./model.pt
```
- to visualze word vectors:
```
python cbow.py ./data --testmodel=./model.pt --visualize
```

### Experiments
1. with 1000 sentences
```
Neighbors of 'mother':
842. mother
916. drum
500. brother
1192. watches
298. daughter
113. favorite
1255. height
197. son
1228. explains
1245. identity
Neighbors of 'father':
364. father
500. brother
175. brothers
916. drum
1181. blessed
478. portrayal
313. performance
1143. owned
530. house
298. daughter
```
2. with 10000 sentences, embedding size = 300
```
Neighbors of 'mother':
317. mother
5163. tender
1316. magdalene
434. father
8168. establishing
8614. blond
200. son
8890. heel
8921. resurection
8443. plotting
485. gift
5176. tenderness
6696. imprint
502. wife
2795. sorrow
2070. nose
8521. dragons
5964. mama
7142. resonance
1438. followers
Neighbors of 'father':
434. father
9127. sentencing
317. mother
9262. foster
485. gift
5789. impressions
4187. rabbi
3984. grave
6899. defeo
3942. identity
657. friend
8785. driver
4611. sending
3234. earthly
6460. upbringing
1581. mission
4922. glowing
655. husband
5246. prays
9166. gap
Neighbors of 'titanic':
6218. titanic
8374. ants
3688. ruthless
3401. breathtaking
4402. engels
4855. liberals
7625. loyal
9018. aweful
1643. girls
6831. judea
9323. frozen
6242. drunken
1992. thousands
4029. corrupt
9189. suspiciously
2606. brown
8594. frog
4946. cowboy
9433. boudicca
```
3. with 40000 sentences, embedding size = 300
```
Neighbors of 'titanic':
3516. titanic
18637. substandard
11397. applauded
7593. afterward
15353. theatrics
6804. publicity
12094. underbelly
17163. sympathizer
10826. bratty
20763. ranged
7550. citizenship
12121. heroin
5705. reluctantly
14814. marxist
16719. patriarchal
16521. floods
19738. dyke
`

