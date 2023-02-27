<?php

namespace sajjadh47;

/**
 * bpe is short for Byte Pair Encoder. It translates arbitrary utf-8 strings into
 * sequences of integers, where each integer represents small chunks of commonly
 * occuring characters. This implementation is based on openai's gpt2 encoder.py:
 * https://github.com/openai/gpt-2/blob/master/src/encoder.py
*/
class BPE
{    
    private $encoder_remote_file  = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json';
    
    private $vocab_remote_file    = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe';
    
    /**
     * Downloads OPEN_API models encoder.json and vocab.bpe remotely if not exists locally
     * and handles caching of these files.
     */
    public function __construct()
    {
        $this->cache_dir            = dirname( __FILE__ ) . DIRECTORY_SEPARATOR . '.cache';
        
        $this->vocab_local_file     = $this->cache_dir . DIRECTORY_SEPARATOR . 'vocab.bpe';
        
        $this->encoder_local_file   = $this->cache_dir . DIRECTORY_SEPARATOR . 'encoder.json';
        
        // Create the cache directory if not exists locally
	    if ( ! file_exists( $this->cache_dir ) )
	    {
	    	mkdir( $this->cache_dir );
	    }

	    # load encoder.json that has the raw mappings from token -> bpe index
	    if ( ! file_exists( $this->encoder_local_file ) )
        {
            $this->dump_remote_file_content( $this->encoder_local_file, $this->encoder_remote_file );
        }

        $this->encoder = json_decode( file_get_contents( $this->encoder_local_file ), true );

        if ( ! $this->encoder && count( $this->encoder ) !== 50257 ) # 256 individual byte tokens, 50,000 merged tokens, and 1 special <|endoftext|> token
        {
            throw new Exception( "{$this->encoder_local_file} content error!!" );
        }

        # load vocab.bpe that contains the bpe merges, i.e. the bpe tree structure
        if ( ! file_exists( $this->vocab_local_file ) )
        {
            $this->dump_remote_file_content( $this->vocab_local_file, $this->vocab_remote_file );
        }

        # light postprocessing: strip the version on first line and the last line is a blank
        $this->bpe_merges = array_map( function( $bpe )
        {
            return implode( ',', explode( ' ', $bpe ) );
        
        }, array_slice( explode( "\n", file_get_contents( $this->vocab_local_file ) ), 1, -1 ) );

        if ( count( $this->bpe_merges ) !== 50000 ) # 50,000 merged tokens
        {
            throw new Exception( "{$this->vocab_local_file} content error!!" );
        }

        $this->bpe_ranks = array_flip( $this->bpe_merges );

        # byte encoder
        $this->byte_encoder = $this->bytes_to_unicode();
        
        # byte decoder
        $this->byte_decoder = array_flip( $this->byte_encoder );
        
        // # bpe token encoder/decoder
        $this->decoder = array_flip( $this->encoder );
        
        // # the splitting pattern used for pre-tokenization
        // ok so what is this regex looking for, exactly?
        // - the vertical bars | is OR, so preg_split will chunkate text as the pieces match, from left to right
        // - '\'s' would split up things like Andrej's -> (Andrej, 's)
        // - ' ?\p{L}': optional space followed by 1+ unicode code points in the category "letter"
        // - ' ?\p{N}': optional space followed by 1+ unicode code points in the category "number"
        // - ' ?[^\s\p{L}\p{N}]+': optional space, then 1+ things that are NOT a whitespace, letter or number
        // - '\s+(?!\S)': 1+ whitespace characters (e.g. space or tab or etc) UNLESS they are followed by non-whitespace
        //                so this will consume whitespace characters in a sequence but exclude the last whitespace in
        //                that sequence. that last whitespace has the opportunity to then match the optional ' ?' in
        //                earlier patterns.
        // - '\s+': 1+ whitespace characters, intended probably to catch a full trailing sequence of whitespaces at end of string
        // So TLDR:
        // - we are special casing a few common apostrophe constructs ('s, 't, 're, ...) and making those into separate tokens
        // - we then separate out strings into consecutive chunks of 1) letters, 2) numbers, 3) non-letter-numbers, 4) whitespaces
        
        $this->regex_pattern = "/(?:\\\\u[a-f0-9]+)+|\'[stdm]|\'[rv]e|\'ll| ?\p{L}+| ?\p{N}+| ?(?!\\\\u[a-f0-9]+\b)[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/m";
        
        $this->cache = [];
    }

    /**
     * this function uses self.bpe_ranks to iteratively merge all the possible bpe tokens
     * up the tree. token is a string of one individual 'word' (after regex tokenization)
     * and after byte encoding, e.g. 'Ġthere'.
     *
     * @return array
     */
    private function bpe( $token )
    {
        # token is a string of one individual 'word', after byte encoding, e.g. 'Ġthere'

        # memoization, for efficiency
        if ( in_array( $token, $this->cache ) )
        {
            return $this->cache[$token];
        }

        $word = mb_str_split( $token ); # individual characters that make up the token
        
        $pairs = $this->get_pairs( $word ); # get all bigrams

        if( ! $pairs )
        {
            return $token;
        }

        while( true )
        {
            # find the next lowest rank bigram that can be merged
            $minPairs = [];
            
            array_map( function( $pair ) use ( &$minPairs )
            {
                $pair = implode( ',', $pair );
                
                $index = ! isset( $this->bpe_ranks[$pair] ) ? 10e10 : $this->bpe_ranks[$pair];

                $minPairs[$index] = $pair;
            
            }, $pairs );

            $bigram = $minPairs[min( array_map( function( $x )
                {
                    return intval( $x );
                
                }, array_keys( $minPairs ) )
            ) ];

            if ( ! in_array( $bigram, $this->bpe_ranks ) )
            {
                break; # no more bigrams are eligible to be merged
            }

            # we will now replace all occurences of (first, second) in the list of current
            # words into one merged token first_second, in the output list new_words
            $first = $bigram[0];
            
            $second = $bigram[1];
            
            $new_word = [];
            
            $i = 0;

            while ( $i < count( $word ) )
            {
                # find the next occurence of first in the sequence of current words
                $j = array_search( $word, array_slice( $first, $i, null, true ) );
                
                if ( $j === false )
                {
                    $new_word = array_merge( $new_word, array_slice( $word, $i ) );
                    
                    break;
                }
                  
                $new_word = array_merge( $new_word, array_slice( $word, $i, $j ) );
                
                $i = $j;

                # if this occurence is also followed by second, then merge them into one
                if ( $word[$i] === $first && $i < count( $word ) - 1 && $word[$i + 1] === $second )
                {
                    $new_word[] = $first . $second;
                    
                    $i = $i + 2;
                }
                else
                {
                    $new_word[] = $word[$i];
                    
                    $i = $i + 1;
                }
            }

            # all occurences of (first, second) have been merged to first_second
            $word = $new_word;
            
            if ( count( $word ) === 1 )
            {
                break;
            }
            else
            {
                $pairs = $this->get_pairs( $word );
            }
        }

        # concat all words into a string, and use ' ' as the separator. Note that
        # by now all characters have been byte encoded, guaranteeing that ' ' is
        # not used in the actual data and is a 'special' delimiter character
        $word = implode( ' ', $word );

        # cache the result and return
        $this->cache[$token] = $word;
        
        return $word;
    }

    /**
     * Encode emoji in text
     * @param string $text text to encode
     */
    public static function EmojiEncode( $text )
    {
        return self::convertEmoji( $text, "ENCODE" );
    }

    /**
     * Decode emoji in text
     * @param string $text text to decode
     */
    public static function EmojiDecode( $text )
    {
        return self::convertEmoji( $text, "DECODE" );
    }

    private static function convertEmoji( $text, $op )
    {
        if( $op == "ENCODE" )
        {
            return preg_replace_callback( '/([0-9|#][\x{20E3}])|[\x{00ae}|\x{00a9}|\x{203C}|\x{2047}|\x{2048}|\x{2049}|\x{3030}|\x{303D}|\x{2139}|\x{2122}|\x{3297}|\x{3299}][\x{FE00}-\x{FEFF}]?|[\x{2190}-\x{21FF}][\x{FE00}-\x{FEFF}]?|[\x{2300}-\x{23FF}][\x{FE00}-\x{FEFF}]?|[\x{2460}-\x{24FF}][\x{FE00}-\x{FEFF}]?|[\x{25A0}-\x{25FF}][\x{FE00}-\x{FEFF}]?|[\x{2600}-\x{27BF}][\x{FE00}-\x{FEFF}]?|[\x{2600}-\x{27BF}][\x{1F000}-\x{1FEFF}]?|[\x{2900}-\x{297F}][\x{FE00}-\x{FEFF}]?|[\x{2B00}-\x{2BF0}][\x{FE00}-\x{FEFF}]?|[\x{1F000}-\x{1F9FF}][\x{FE00}-\x{FEFF}]?|[\x{1F000}-\x{1F9FF}][\x{1F000}-\x{1FEFF}]?/u', array( 'self', "encodeEmoji" ), $text );
        }
        else
        {
            return preg_replace_callback( '/(\\\u[0-9a-f]{4})+/', array( 'self', "decodeEmoji" ), $text );
        }
    }

    private static function encodeEmoji( $match )
    {
        return str_replace( array( '[', ']', '"' ), '', json_encode( $match ) );
    }

    private static function decodeEmoji( $text )
    {
        if( ! $text ) return '';
        
        $text = $text[0];
        
        $decode = json_decode( $text, true );
        
        if( $decode ) return $decode;
        
        $text = '["' . $text . '"]';
        
        $decode = json_decode( $text );
        
        if( count( $decode ) == 1 )
        {
           return $decode[0];
        }
        
        return $text;
    }

    /**
     * string goes in, list of integers comes out
     *
     * @return array
     */
    public function encode( $text )
    {
        $bpe_idx = [];

        # pre-tokenize the input text into string tokens (words, roughly speaking)
        preg_match_all( $this->regex_pattern, $this->EmojiEncode( $text ), $tokens );
        
        # process each token into BPE integers
        if ( isset( $tokens[0] ) )
        {
            foreach ( $tokens[0] as $token )
            {
                # encode the token as utf-8
                $token_bytes = iconv( mb_detect_encoding( $token, mb_detect_order(), true ), "UTF-8", $token );
                
                # translate all bytes to their unicode string representation and flatten
                $token_translated = '';

                foreach ( mb_str_split( $token_bytes ) as $token_byte )
                {
                    $token_translated .= $this->byte_encoder[mb_ord( $token_byte )];
                }
                
                // # perform all the applicable bpe merges according to $this->bpe_ranks
                $token_merged = implode( '', mb_split( "\s", $this->bpe( $token_translated ) ) );

                // # translate all bpe tokens to integers
                if ( ! isset( $this->encoder[$token_merged] ) )
                {
                    foreach ( mb_str_split( $token_merged ) as $unknown_byte )
                    {
                        $bpe_idx[] = $this->encoder[$unknown_byte];
                    }
                }
                else
                {
                    $bpe_idx[] = $this->encoder[$token_merged];
                }
            }
        }
        
        return $bpe_idx;
    }

    /**
     * list of integers comes in, string comes out
     *
     * @return string
     */
    public function decode( $bpe_idx )
    {
        // # inverse map the integers to get the tokens
        $tokens_merged = [];
        
        foreach ( $bpe_idx as $token )
        {
            if ( isset( $this->decoder[$token] ) )
            {
                $tokens_merged[] = $this->decoder[$token];
            }    
        }
        
        // # inverse the byte encoder, e.g. recovering 'Ġ' -> ' ', and get the bytes
        $tokens_flat = mb_str_split( implode( '', $tokens_merged ) );
        
        $tokens_bytes = [];

        foreach ( $tokens_flat as $token_flat )
        {
            if ( isset( $this->byte_decoder[$token_flat] ) )
            {
                $tokens_bytes[] = chr( $this->byte_decoder[$token_flat] );
            }
            else
            {
                $tokens_bytes[] = $token_flat;
            }
        }
        
        // # recover the full utf-8 string & if needed decode any emoji encoded data
        $text = $this->EmojiDecode( implode( '', $tokens_bytes ) );
        
        return $text;
    }

    /**
     * downloads remote_file to local_file if necessary
     */
    private function dump_remote_file_content( $local_file, $remote_file )
    {
        file_put_contents( $local_file, file_get_contents( $remote_file ) );
    }

    /**
     * Every possible byte (really an integer 0..255) gets mapped by OpenAI to a unicode
     * character that represents it visually. Some bytes have their appearance preserved
     * because they don't cause any trouble. These are defined in list bs. For example:
     * chr(33) returns "!", so in the returned dictionary we simply have d[33] -> "!".
     * However, chr(0), for example, is '\x00', which looks ugly. So OpenAI maps these
     * bytes, into new characters in a range where chr() returns a single nice character.
     * So in the final dictionary we have d[0] -> 'Ā' instead, which is just chr(0 + 2**8).
     * In particular, the space character is 32, which we can see by ord(' '). Instead,
     * this function will shift space (32) by 256 to 288, so d[32] -> 'Ġ'.
     * So this is just a simple one-to-one mapping of bytes 0..255 into unicode characters
     * that "look nice", either in their original form, or a funny shifted character
     * like 'Ā', or 'Ġ', etc.
     *
     * @return array
     */
    private function bytes_to_unicode()
    {
        # the 188 integers that render fine in their original form and need no shifting
        $cs = $bs = [ ...range( mb_ord( "!" ), mb_ord( "~" ) ), ...range( mb_ord( "¡" ), mb_ord( "¬" ) ), ...range( mb_ord( "®" ), mb_ord( "ÿ" ) ) ];
        
        # all integers in cs will simply map to chr(c) in the output
        # now get the representations of the other 68 integers that do need shifting
        # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
        $n = 0;
        
        foreach ( range( 0, 2**8 - 1 ) as $b )
        {    
            if ( ! in_array( $b, $bs ) )
            {
                # if this byte is "ugly" then map it to the next available "nice" character
                array_push( $bs, $b );
                
                array_push( $cs, 2**8 + $n );
                
                $n += 1;
            }
        }

        return array_combine( $bs, array_map( 'mb_chr', $cs ) );
    }

    /**
     * Return all bigrams as an array, of consecutive elements in the iterable word.
     *
     * @return array
     */
    private function get_pairs( $word )
    {
        $pairs = [];
        
        $prev_char = $word[0];
        
        for( $i = 1; $i < count( $word ); $i++ )
        {
            $char = $word[$i];
            
            $pairs[] = [ $prev_char, $char ];
            
            $prev_char = $char;
        }
        
        return $pairs;
    }
}
