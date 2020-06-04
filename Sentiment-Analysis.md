---
title: "Sentiment Analysis (with possible pre-workshop typos)"
author: "Dave Campbell"
date: "2020-06-04Part2.1"
---

Libraries used in this section
```r

library(tidyverse)
library(tidytext)
library(dplyr)
library(genius)
library(stringr)
library(janitor)

```

# Obtaining Song lyrics


We can obtain song lyrics using library(genius), which is a API to [genius.com](https://genius.com/Rick-astley-never-gonna-give-you-up-lyrics).  I have the best luck finding lyrics by first looking for the precise spelling on genius.com, for example determining if we should use "Beatles" or "The Beatles".



Say you want the lyrics for a song that's [really catchy](https://www.youtube.com/watch?v=dQw4w9WgXcQ).
The main function takes an artist and a song:
```r 
Artist = "Frank Turner" 
Song = "Be More Kind"
genius_lyrics(artist=Artist ,song= Song)
```

We can also find all the songs from an album: 
```r
genius_tracklist(artist="The Clash", album = "London Calling")

```

Today we are interested in obtaining all lyrics from all songs on a album.  Note that the API is fragile and sometimes it won't give us all that we _desire_.


```r

Artist = "Bob Dylan"
Album = "Desire"

Lyrics = genius_album(artist = Artist,album = Album, info = "all")

Lyrics$track_title %>% table

```


Some days it's better to loop over song titles to get song lyrics for the album.

```r
Artist = "Bob Dylan"
Album = "Desire"
tracklist = genius_tracklist(artist=Artist, album = Album)

#Loop over all tracks on an album:
LyricsDesire = NULL
for(songNumber in 1:dim(tracklist)[1]){
   songlength = 1
   print(tracklist$track_title[songNumber])
   counter = 0 # avoid infinite loops for instrumental songs
   while(songlength<=1 & counter<10){
        counter = counter +1
        (NewSong = genius_lyrics(artist=Artist ,song= tracklist$track_title[songNumber]))
        Sys.sleep(1)
        songlength = dim(NewSong)[1]
   }
   print(songlength)
   LyricsDesire = rbind(LyricsDesire,NewSong)
}

LyricsDesire$track_title %>% table
```

I've included a System Sleep call since this will slow down the calls to the genius webpage.  This is a necessary tool when web scraping, even through an API.  Here that step isn't enough so I also look for song length returned by the function call and keep trying as long as it is empty for up to 10 tries.  Sometimes an artist puts an instrumental song or doesn't provide lyrics, so the _counter_ for the number of tries avoids infinite loops. 



# Sentiment Analysis for music

Did Justin Bieber grow up a little over the years?  Let's find out!  First we will obtain some lyrics for his first and latest albums.  Then we will map the lyrics into sentiment categories and compare the distributions thereof between albums.

```r
Artist = "Justin Bieber"
Album = "My world ep"
tracklist = genius_tracklist(artist=Artist, album = Album)
#All lyrics from an album:
LyricsJB1 = NULL
for(songNumber in 1:dim(tracklist)[1]){
   songlength = 1
   print(tracklist$track_title[songNumber])
   counter = 0 # avoid infinite loops for instrumental songs
   while(songlength<=1 & counter<10){
        counter = counter +1
        (NewSong = genius_lyrics(artist=Artist ,song= tracklist$track_title[songNumber]))
        Sys.sleep(1)
        songlength = dim(NewSong)[1]
   }
   LyricsJB1 = rbind(LyricsJB1,NewSong)
}


Album = "Changes"
tracklist = genius_tracklist(artist=Artist, album = Album)
tracklist$track_title[15] = "Thats What Love Is"
tracklist$track_title[12] = "eta"
LyricsJB2 = NULL
for(songNumber in 1:dim(tracklist)[1]){
   songlength = 1
   print(tracklist$track_title[songNumber])
   counter = 0 # avoid infinite loops for instrumental songs
   while(songlength<=1 & counter<10){
        counter = counter +1
        (NewSong = genius_lyrics(artist=Artist ,song= tracklist$track_title[songNumber]))
        Sys.sleep(1)
        songlength = dim(NewSong)[1]
   }
   LyricsJB2 = rbind(LyricsJB2,NewSong)
}


LyricsJB1$track_title %>% table
LyricsJB2$track_title %>% table
```

Note that we lost the "Yummy (Summer Walker Remix)", but we're not sad about that.

## Sentiment Analysis

The idea is to  match terms to a sentiment lexicon.  These have been built for a specific purpose, but hopefully such a sentiment will be useful for us.

There is no way of separating out data cleaning from data analysis.

Sentiment Analysis is all about comparing words in a document to words in a sentiment list. The best is to build your own lexicon that suits your needs, but that's expensive and slow. There are 3 popular lexicons:

- AFINN from Finn Årup Nielsen,
- bing from Bing Liu and collaborators, and
- nrc from Saif Mohammad and Peter Turney at the National Research Council of Canada.

_AFINN_ gives words a score between -5 and +5 rating its severity of positive or negative sentiment.
```r
tidytext::get_sentiments("afinn")
hist(get_sentiments("afinn")$value)
```

_bing_ gives a binary value of positive or negative. Neutral words are not in the list.
```r
tidytext::get_sentiments("bing")
table(get_sentiments("bing")$sentiment)
```

_nrc_ puts each word into a sentiment category.
```r
tidytext::get_sentiments("nrc")
table(get_sentiments("nrc")$sentiment)
```


Note that each lexicon has a different length and words evolve in meaning over time.  Sick used to be bad, then it was good, now it's so bad that we stay home most of the time.



Back to sentiments, let's count the _nrc_ category occurences:

```r
Lyrics1 = LyricsJB1 %>% mutate(album= "My world")
Lyrics2 = LyricsJB2 %>% mutate(album= "Changes")
Lyrics = rbind(Lyrics1,Lyrics2) %>%
         unnest_tokens(output = word,input = lyric, token = "words") %>%
         inner_join(get_sentiments("nrc") )
#Count the occurrence with each album
Sents = Lyrics %>%group_by(album)%>%count(sentiment)

Sents %>% summarise(sum(n))

Sents %>% filter(!(sentiment =="positive"|sentiment=="negative")) %>%
ggplot(aes(fill=sentiment, y=n, x=album)) + 
    geom_bar(position="fill", stat="identity")
```

# Statistical Test


Consider the Null Hypothesis that Justin Bieber has the same sentiment distribution between 'My World' and 'Changes'.  The alternative is that there is an unspecified difference in distribution.  This can be tested through a Chi-Square test for categorical distributions with $N_r$ rows and $N_c$ columns.
With observed counts $O_{ij}$ in row $i$ and column $j$ of the table, the Expected counts are the data assuming the only difference is the total count; 
\[E_{ij}= (\mbox{row i total*column j total}) / N_{total}\]
This gives the test statistic:

\[X = \sum_{i=1}^{N_r}\sum_{j=1}^{N_c}\frac{(O_{ij}-E_{ij})^2}{E_{ij}}\sim \chi^2_{(N_r-1)*(N_c-1)}\]

*Note1* that we will remove the categories "positive" and "negative" since these are supersets of the other _nrc_ categories.

*Note2* The basic assumption is that these songs are 'bags of words' that Bieber randomly samples into lyrics.  Essentially Bieber has some sentiment distribution at a given time and the lyrics are a random sample thereof.

```r
#Cross-tab table:

MyTable = janitor::tabyl(Lyrics%>% filter(!(sentiment =="positive"|sentiment=="negative")) %>%
      group_by(album)   ,sentiment,album)
```


The janitor library makes it easy to build tables that look nice and can format data with row or column percents and counts.

```r
MyTable %>% adorn_totals("row")
MyTable %>% adorn_percentages("col")%>%  adorn_ns()

MyTable %>% adorn_percentages("row")
MyTable %>% adorn_percentages("col")
MyTable %>% adorn_percentages("all")%>%  adorn_ns()


#Chi-Square test:
chisq.test(MyTable)
```


There appears to be strong evidence of a change in distribution of sentiments between Justin Bieber Albums.





# Going furhter: Improving the lexicon by considering negation

Using a lexicon, consider the phrase: 
- "my homemade bread is not bad"  
A lexicon based approach would see the word _bad_ and consider the sentence to be negative, whereas _not bad_ is actually pretty good all things considered.

To keep it simple consider just the _bing_ lexicon and first look for negation.
First we will split the lyrics into bigrams, then find places where the first word is a common negation.

The bigram object _Biebgrams_ splits the text into sliding groupings of two word pairs. From there we split apart the bigram and perform sentiment analysis on just the second word.  To get around this we re introduce a NA at the start of each line.  Otherwise the first word of each column (here appearing in column "preword") will be ignored by the sentiment analysis.  Hopefully this makes more sense when you look at the steps.

```r
# split into bigrams, but keep the original text to make every step clear
Biebgrams 


Bieber = rbind(Lyrics1,Lyrics2) %>%
       mutate(originalline = lyric) 

# glue NA to the beginning of each string.  The result is that now all of teh original words will be used in the sentiment lexicon replacement.
Bieber$lyric = stringr::str_replace_all(Bieber$lyric, pattern="^", replacement = "NA ")
       

# Split into bigrams, separate apart the bigram into "PreWord" and "word" and perform lexicon sentiment replacement on the second term ("word") while keeping way too many interim steps for illustration:

Bieber   =  Bieber %>%
            unnest_tokens(output = bigram,input = lyric, token = "ngrams", n=2) %>%
            mutate(originalbigram = bigram) %>%
            separate(bigram, c("PreWord", "word"), sep = " ") %>%
            inner_join(get_sentiments("bing") ) 



# find bigrams where the first word might negate the second.
Bieber %>% filter(PreWord %in% c("not","isn't","no"))
# now consider which should be changed.


Bieber[which(Bieber$originalbigram=="no patience"),"sentiment"] = "negative"    
Bieber[which(Bieber$originalbigram=="no approval"),"sentiment"] = "negative"
Bieber[which(Bieber$originalbigram=="not right") ,"sentiment"] = "negative"

Bieber[which(Bieber$originalbigram=="no wrong"),"sentiment"] = "positive"
Bieber[which(Bieber$originalbigram=="not trouble"),"sentiment"] = "positive"

Bieber %>% filter(PreWord %in% c("not","isn't","no"))

```

If the text is expected to have a small number of typos of typos the _amatch_ function from _library(stringdist)_ will look for an approximate match rather than an exact match for strings.



 