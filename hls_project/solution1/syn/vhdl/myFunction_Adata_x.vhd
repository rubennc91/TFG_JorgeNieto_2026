-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
--
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity myFunction_Adata_x_ram is 
    generic(
            DWIDTH     : integer := 32; 
            AWIDTH     : integer := 6; 
            MEM_SIZE    : integer := 43
    ); 
    port (
          addr0     : in std_logic_vector(AWIDTH-1 downto 0); 
          ce0       : in std_logic; 
          d0        : in std_logic_vector(DWIDTH-1 downto 0); 
          we0       : in std_logic; 
          q0        : out std_logic_vector(DWIDTH-1 downto 0);
          clk        : in std_logic 
    ); 
end entity; 


architecture rtl of myFunction_Adata_x_ram is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
shared variable ram : mem_array := (
    0 => "10111111100000000000000000000000", 
    1 => "00111111100000000000000000000000", 
    2 => "10111111100000000000000000000000", 
    3 => "00111111100000000000000000000000", 
    4 => "10111111100000000000000000000000", 
    5 => "00111000110100011011011100010111", 
    6 => "00111111100000000000000000000000", 
    7 => "10111111100000000000000000000000", 
    8 => "00111111100000000000000000000000", 
    9 => "10111111100000000000000000000000", 
    10 => "00111111100000000000000000000000", 
    11 => "10111111100000000000000000000000", 
    12 => "00111000110100011011011100010111", 
    13 => "00111111100000000000000000000000", 
    14 to 16=> "10111111100000000000000000000000", 
    17 => "00111111100000000000000000000000", 
    18 => "10110111000000101111010011011110", 
    19 => "10111011101000111101011100001010", 
    20 => "00111111100000000000000000000000", 
    21 => "10110110111110100110111011010100", 
    22 => "00111000110100011011011100010111", 
    23 => "10110111000000101111010011011110", 
    24 => "10111011101000111101011100001010", 
    25 => "00110111000000101111010011011110", 
    26 => "00111011101000111101011100001010", 
    27 => "10110111000000101111010011011110", 
    28 => "10111011101000111101011100001010", 
    29 => "00110001101010111100110001110111", 
    30 => "00111000110100011011011100010111", 
    31 => "10110110111110100110111011010100", 
    32 => "00110110111110100110111011010100", 
    33 => "10110110111110100110111011010100", 
    34 => "00111000110100011011011100010111", 
    35 => "10110111000000101111010011011110", 
    36 => "10111011101000111101011100001010", 
    37 => "00110111000000101111010011011110", 
    38 => "00111011101000111101011100001010", 
    39 => "00110001101010111100110001110111", 
    40 => "00111000110100011011011100010111", 
    41 => "10110110111110100110111011010100", 
    42 => "00110110111110100110111011010100" );


begin 


memory_access_guard_0: process (addr0) 
begin
      addr0_tmp <= addr0;
--synthesis translate_off
      if (CONV_INTEGER(addr0) > mem_size-1) then
           addr0_tmp <= (others => '0');
      else 
           addr0_tmp <= addr0;
      end if;
--synthesis translate_on
end process;

p_memory_access_0: process (clk)  
begin 
    if (clk'event and clk = '1') then
        if (ce0 = '1') then 
            q0 <= ram(CONV_INTEGER(addr0_tmp));
            if (we0 = '1') then 
                ram(CONV_INTEGER(addr0_tmp)) := d0; 
            end if;
        end if;
    end if;
end process;


end rtl;

Library IEEE;
use IEEE.std_logic_1164.all;

entity myFunction_Adata_x is
    generic (
        DataWidth : INTEGER := 32;
        AddressRange : INTEGER := 43;
        AddressWidth : INTEGER := 6);
    port (
        reset : IN STD_LOGIC;
        clk : IN STD_LOGIC;
        address0 : IN STD_LOGIC_VECTOR(AddressWidth - 1 DOWNTO 0);
        ce0 : IN STD_LOGIC;
        we0 : IN STD_LOGIC;
        d0 : IN STD_LOGIC_VECTOR(DataWidth - 1 DOWNTO 0);
        q0 : OUT STD_LOGIC_VECTOR(DataWidth - 1 DOWNTO 0));
end entity;

architecture arch of myFunction_Adata_x is
    component myFunction_Adata_x_ram is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            we0 : IN STD_LOGIC;
            d0 : IN STD_LOGIC_VECTOR;
            q0 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    myFunction_Adata_x_ram_U :  component myFunction_Adata_x_ram
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        we0 => we0,
        d0 => d0,
        q0 => q0);

end architecture;


