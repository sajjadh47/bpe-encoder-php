## About
BPE Encoder Decoder for GPT-2 / GPT-3 Implemented In PHP. This is a PHP implementation of OpenAI's original python encoder/decoder which can be found [here](https://github.com/openai/gpt-2/blob/master/src/encoder.py). It follows 99% exact OpenAI's Python's implementation algorithms to acquire the speed and multi byte encodings.

## Install with composer

```
composer require sajjadh47/bpe-encoder-php
```

## Usage

Minimum PHP Version For This Package Is >= 7.4 With `mb_*` [functions](https://www.php.net/manual/en/book.mbstring.php) enabled for multi byte encodings.

```php
<?php

include 'vendor/autoload.php';

use SajjadHSagor\BPE;

$BPE = new BPE();

$encoded = $BPE->encode( "Hello!! I'm Sajjad Hossain Sagor. It's 2023, Nice To Meet You. What's Up? :) 🤗" );

print_r( $encoded );

//Outputs
Array
(
    [0] => 15496
    [1] => 3228
    [2] => 314
    [3] => 1101
    [4] => 220
    [5] => 50
    [6] => 64
    [7] => 73
    [8] => 73
    [9] => 64
    [10] => 67
    [11] => 220
    [12] => 39
    [13] => 78
    [14] => 82
    [15] => 82
    [16] => 64
    [17] => 72
    [18] => 77
    [19] => 220
    [20] => 50
    [21] => 64
    [22] => 70
    [23] => 78
    [24] => 81
    [25] => 13
    [26] => 632
    [27] => 338
    [28] => 220
    [29] => 17
    [30] => 15
    [31] => 17
    [32] => 18
    [33] => 11
    [34] => 18460
    [35] => 1675
    [36] => 21167
    [37] => 921
    [38] => 13
    [39] => 1867
    [40] => 338
    [41] => 3205
    [42] => 30
    [43] => 14373
    [44] => 3467
    [45] => 463
    [46] => 5999
    [47] => 68
    [48] => 59
    [49] => 4185
    [50] => 1558
)

echo $BPE->decode( $encoded );

//Outputs
Hello!! I'm Sajjad Hossain Sagor. It's 2023, Nice To Meet You. What's Up? :) 🤗

```
