<?php

class Mao {
    
    public function __construct($key) {
        $salt = 'what happened in vegas stays in vegas';
        $this->key = hash('sha256', $key . $salt, true);
        $this->method = 'AES-256-OFB';
    }

    public function encrypt($data) {
        $iv = random_bytes(16);
        $data = json_encode($data);
        $padding = 16 - (strlen($data) % 16);
        $data = str_repeat(chr($padding), $padding) . $data;
        $cipher = openssl_encrypt($data, $this->method, $this->key, OPENSSL_RAW_DATA, $iv);
        var_dump('cipher = '. bin2hex($cipher) . "\n");
        return base64_encode($iv . $cipher);
    }

    public function decrypt($cipher) {
        $cipher = base64_decode($cipher);
        $iv = substr($cipher, 0, 16);
        $cipher = substr($cipher, 16);
        $data = openssl_decrypt($cipher, $this->method, $this->key, OPENSSL_RAW_DATA, $iv);
        $data = substr($data, 0, ord($data[strlen($data) - 1]));
        return json_decode($data, true);
    }
}

