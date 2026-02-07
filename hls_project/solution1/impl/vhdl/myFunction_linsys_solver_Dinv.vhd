-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
--
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity myFunction_linsys_solver_Dinv_ram is 
    generic(
            DWIDTH     : integer := 32; 
            AWIDTH     : integer := 6; 
            MEM_SIZE    : integer := 34
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


architecture rtl of myFunction_linsys_solver_Dinv_ram is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
shared variable ram : mem_array := (
    0 to 1=> "10111110000101000101101000110001", 
    2 => "11000011000100001110000000010101", 
    3 => "00111000110011101001000111001001", 
    4 => "00110111001001100001100000101101", 
    5 => "10111110000101000101101000110001", 
    6 => "11000011000100001010101001101101", 
    7 => "00111000110011101001000111001001", 
    8 => "11000011000011001101101000110100", 
    9 => "10111110000101000101101000110001", 
    10 => "11000011000100001110000000010101", 
    11 => "00111011111000100010110110000111", 
    12 => "11000011000100001110000000010101", 
    13 => "10111110000101000101101000110001", 
    14 => "00111011111000100010110110000111", 
    15 to 16=> "10111110000101000101101000110001", 
    17 => "11000011000100001110000000010101", 
    18 => "00111000110011101001000111001001", 
    19 => "11000011000100001110000000010101", 
    20 => "00111011111000100010110110000111", 
    21 => "00110110010111010111010110010001", 
    22 => "00111000110100011000000101100111", 
    23 => "00111100001000111101011010011111", 
    24 => "11000010011011001010011011001101", 
    25 => "11000011000011101011110100001100", 
    26 => "00111100100010100111011011100111", 
    27 => "11000011000011001101101000110110", 
    28 => "11000010000000100100001011110010", 
    29 => "01001001001110000010100111111001", 
    30 => "01000101100100101000011001101001", 
    31 => "01000110100000111010011000000101", 
    32 => "01001001010000111111111000001000", 
    33 => "10111110000101000101100110010000" );


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

entity myFunction_linsys_solver_Dinv is
    generic (
        DataWidth : INTEGER := 32;
        AddressRange : INTEGER := 34;
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

architecture arch of myFunction_linsys_solver_Dinv is
    component myFunction_linsys_solver_Dinv_ram is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            we0 : IN STD_LOGIC;
            d0 : IN STD_LOGIC_VECTOR;
            q0 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    myFunction_linsys_solver_Dinv_ram_U :  component myFunction_linsys_solver_Dinv_ram
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        we0 => we0,
        d0 => d0,
        q0 => q0);

end architecture;


