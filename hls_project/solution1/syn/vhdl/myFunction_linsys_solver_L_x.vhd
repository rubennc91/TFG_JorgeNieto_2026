-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
--
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity myFunction_linsys_solver_L_x_ram is 
    generic(
            DWIDTH     : integer := 32; 
            AWIDTH     : integer := 6; 
            MEM_SIZE    : integer := 57
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


architecture rtl of myFunction_linsys_solver_L_x_ram is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
shared variable ram : mem_array := (
    0 => "00110101100101111010101010000001", 
    1 => "00110101100100001111010010000100", 
    2 => "00111010001111011110001110100110", 
    3 => "01000011000100001110000000010101", 
    4 => "00111000110011101011011101011110", 
    5 => "10110111001001111001101010111001", 
    6 => "00111010001111011110001110100110", 
    7 => "11000011000100001010101001101110", 
    8 => "10111100011011010000010100110001", 
    9 => "10111000110011101011011101011110", 
    10 => "00110101101111101001011011010100", 
    11 => "10111100011001101100010110111101", 
    12 => "10111001010100101001101101000011", 
    13 => "00111010001111011110001110100110", 
    14 => "10111010001111011110001110100110", 
    15 => "11000011000100001110000000010101", 
    16 => "10110011010101101011111110010101", 
    17 => "11000011000100001110000000010101", 
    18 => "00111010001111011110001110100110", 
    19 => "10111010001111011110001110100110", 
    20 => "10110011010101101011111110010101", 
    21 => "10110101001111101001011011010100", 
    22 => "10110101100100001111010010000100", 
    23 => "10110101100101111010101010000001", 
    24 => "00110101100100001111010010000100", 
    25 => "00110101100101111010101010000001", 
    26 => "01000011000100001110000000010101", 
    27 => "00111000110011101011011101011110", 
    28 => "01000011000100001110000000010101", 
    29 => "00110101001110010011100001110000", 
    30 => "00111011111000100010111000110011", 
    31 => "10110110010111110111100011110111", 
    32 => "00111000110100011011000110111001", 
    33 => "10111000110100011011000110111001", 
    34 => "10111100001000111101011100000000", 
    35 => "11000010011011001010011011001101", 
    36 => "10111011110000011101110101011011", 
    37 to 38=> "10111100011010011101110011010011", 
    39 => "10110101001111101001011011010100", 
    40 => "00110010101010111100110001110111", 
    41 => "10111100100010100111011100010010", 
    42 => "00111000110100011011000110111001", 
    43 => "00111000110001001100000100111011", 
    44 => "10110101001110111110011110100010", 
    45 => "00110010001010111100110001110111", 
    46 to 47=> "10111011010101010110101110000010", 
    48 => "00110111010101011001001011101111", 
    49 => "00111110011110111000110001000001", 
    50 => "11000000101101000010100010111000", 
    51 => "10111110111110001011011010011110", 
    52 => "10110010001010111100110001110111", 
    53 => "10111101000101011110100001011010", 
    54 => "00110011110101101011111110010101", 
    55 => "00111101100010101000011001001101", 
    56 => "01000000111011101101010001000001" );


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

entity myFunction_linsys_solver_L_x is
    generic (
        DataWidth : INTEGER := 32;
        AddressRange : INTEGER := 57;
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

architecture arch of myFunction_linsys_solver_L_x is
    component myFunction_linsys_solver_L_x_ram is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            we0 : IN STD_LOGIC;
            d0 : IN STD_LOGIC_VECTOR;
            q0 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    myFunction_linsys_solver_L_x_ram_U :  component myFunction_linsys_solver_L_x_ram
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        we0 => we0,
        d0 => d0,
        q0 => q0);

end architecture;


