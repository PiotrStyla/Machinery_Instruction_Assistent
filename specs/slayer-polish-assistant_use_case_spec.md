# Use Case Specification

## Business Problem

Przygotować lokalny, polskojęzyczny model Slayer1B do odpowiadania na jasno sformułowane instrukcje użytkownika, z mniejszą liczbą powtórzeń, lepszą zgodnością z poleceniem i większą poprawnością krótkich odpowiedzi niż model bazowy Slayer1B-1.05.

## Primary Users

Piotr Styła jako twórca i tester lokalnego polskiego modelu; użytkownicy testujący model offline na komputerze bez chmury.

## Success Tenets

- **Zgodność z instrukcją** — na co najmniej 80% przykładów z odłożonego zestawu testowego model wykonuje żądane zadanie i odpowiada w oczekiwanym formacie.
- **Jakość języka polskiego** — odpowiedzi uzyskują średnią ocenę co najmniej 4/5 za poprawność językową i spójność w ślepej ocenie zestawu testowego.
- **Stabilność generowania** — odsetek odpowiedzi z zapętleniem, samymi separatorami lub jawną sprzecznością faktograficzną spada co najmniej o połowę względem modelu bazowego na tym samym zestawie testowym.
